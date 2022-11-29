import spotlight
import datasets
import numpy as np
import argparse
import pickle
import os
import requests
from tqdm.auto import tqdm
from collections import defaultdict
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--dataset', default='openwebtext', choices=['openwebtext','the_pile','roots','wikipedia'], help='dataset to entity link')
    parser.add_argument('--wikipedia_path', default=None, help='path to wikipedia HFDataset if --dataset wikipedia')
    parser.add_argument('--nprocs', default=4, type=int, help='number of processes')
    parser.add_argument('--start', default=0, type=int, help='start index')
    parser.add_argument('--end', default=-1, type=int, help='end index')
    parser.add_argument('--spotlight_port', nargs='+', type=int, default=[2222], help='port serving DBPedia Spotlight')
    parser.add_argument('--confidence', default=0.4, type=float, help='DBPedia confidence parameter')
    parser.add_argument('--support', default=20, type=int, help='DBPedia support parameter')
    parser.add_argument('--auth_token', default=None, type=str, help='Huggingface auth token')
    args = parser.parse_args()
    return args

def annotate(text, port, confidence, support, retry=5):
    endpoint = f'http://localhost:{port}/rest/annotate'
    try:
        return spotlight.annotate(endpoint, text, confidence=confidence, support=support)
    except spotlight.SpotlightException:
        return [{'URI':'__SPOTLIGHT_EXCEPTION__'}]
    except requests.exceptions.ConnectionError:
        return [{'URI':'__CONNECTION_ERROR__'}]
    except requests.exceptions.HTTPError:
        return [{'URI':'__HTTP_ERROR__'}]

def link_fn(example, rank):
    annotations = annotate(example['text'], args.spotlight_port[rank % len(args.spotlight_port)], args.confidence, args.support)
    entity_uris = set([a['URI'] for a in annotations])
    example['entities'] = list(entity_uris)
    return example

def load_dataset(dataset, path, start, end, auth_token):
    print(f'Loading dataset {dataset}')
    if dataset in ['openwebtext', 'the_pile']:
        split = f'[{start}:{end}]' if end != -1 else ''
        return datasets.load_dataset(dataset, split=f'train{split}')
    elif dataset == 'c4':
        split = f'[{start}:{end}]' if end != -1 else ''
        return datasets.load_dataset(dataset, 'en', split=f'train{split}')
    elif dataset == 'roots':
        #Load and concatenate English constituent datasets in ROOTS
        roots_dataset_names = sorted([ds_info.id for ds_info in HfApi().list_datasets(use_auth_token=auth_token) if ds_info.id.startswith('bigscience-data/roots_en_')])
        roots_datasets = []
        for d_name in roots_dataset_names:
            print(f'Loading constituent dataset {d_name}')
            d = datasets.load_dataset(d_name, use_auth_token=not auth_token is None)['train']
            remove = set(d.features.keys()) - {'text'}
            roots_datasets.append(d.remove_columns(remove))
        roots = datasets.concatenate_datasets(roots_datasets)
        if end != -1:
            roots = roots[start:end]
        return roots
    elif dataset == 'wikipedia':
        wiki = datasets.load_from_disk(path)
        if end != -1:
            wiki = wiki[start:end]
        return wiki
             
def main(args):
    entity_map = defaultdict(set)    
    dataset = load_dataset(args.dataset, args.wikipedia_path, args.start, args.end, args.auth_token)
    dataset = dataset.map(link_fn, batched=False, with_rank=True, num_proc=args.nprocs)

    for i, example in enumerate(tqdm(dataset)):
        for entity in example['entities']:
            entity_map[entity].add(i)
    
    for k,v in entity_map.items():
        entity_map[k] = np.array(list(v), dtype=np.int32)

    filename = f'{args.dataset}_{args.start}_{args.end}_entity_map.pkl' if args.end != -1 else f'{args.dataset}_entity_map.pkl'
    with open(os.path.join(args.outdir, filename), 'wb') as f:
        pickle.dump(dict(entity_map), f)

if __name__ == '__main__':
    args = parse_args()
    main(args)
