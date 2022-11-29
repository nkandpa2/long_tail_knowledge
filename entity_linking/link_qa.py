import spotlight
import datasets
import argparse
import functools
import operator
import jsonlines
import multiprocessing as mp
import os
import requests
import random
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--dataset', default='trivia_qa', choices=['trivia_qa', 'nq_open'], help='dataset to entity link')
    parser.add_argument('--dataset_splits', default=['train','validation'], nargs='+', choices=['train', 'validation'], help='dataset split to link')
    parser.add_argument('--dataset_subset', default=None, choices=[None, 'unfiltered.nocontext', 'unfiltered', 'rc.nocontext', 'rc'], help='dataset subset to link (only for trivia qa)')
    parser.add_argument('--nprocs', default=1, type=int, help='number of processes')
    parser.add_argument('--spotlight_ports', nargs='+', default=[2222], type=int, help='port serving DBPedia Spotlight')
    parser.add_argument('--confidence', default=0.4, type=float, help='DBPedia confidence parameter')
    parser.add_argument('--support', default=20, type=int, help='DBPedia support parameter')
    args = parser.parse_args()
    return args

def annotate(text, endpoint, confidence, support, retry=10):
    try:
        return spotlight.annotate(endpoint, text, confidence=confidence, support=support)
    except spotlight.SpotlightException:
        return []
    except requests.exceptions.ConnectionError:
        if retry == 0:
            print('ConnectionError, not retrying')
            return []
        else:
            print(f'ConnectionError, retrying {retry-1}')
            return annotate(text, endpoint, confidence, support, retry=retry-1)
    except requests.exceptions.HTTPError:
        if retry == 0:
            print('HTTPError, not retrying')
            return []
        else:
            print(f'HTTPError, retrying {retry-1}')
            return annotate(text, endpoint, confidence, support, retry=retry-1)


def link_fn(example):
    endpoint = f'http://localhost:{random.choice(args.spotlight_ports)}/rest/annotate'
    question_str = example['question']
    answer_str = question_str + '\n' + '\n'.join(example['answer'])
    return {'q_entities': annotate(question_str, endpoint, args.confidence, args.support),
            'a_entities': [e for e in annotate(answer_str, endpoint, args.confidence, args.support) if e['offset'] >= len(question_str)]}


def load_dataset(args):
    if args.dataset == 'nq_open':
        dataset = datasets.load_dataset(args.dataset)
    elif args.dataset == 'trivia_qa':
        dataset = datasets.load_dataset(args.dataset, args.dataset_subset)
    
    dataset = datasets.concatenate_datasets([dataset[s] for s in args.dataset_splits])
    return dataset


def main(args):
    qa = load_dataset(args)
    dataset = f'{args.dataset}_{args.dataset_subset}' if not args.dataset_subset is None else f'{args.dataset}'
    out_filename = f'{dataset}_{"+".join(args.dataset_splits)}_entities.jsonl'
    with open(os.path.join(args.outdir, out_filename), 'w') as outfile:
        with jsonlines.Writer(outfile) as writer:
            with mp.Pool(args.nprocs) as pool:
                link_itr = pool.imap(link_fn, nq)
                for link in tqdm(link_itr, total=len(nq)):
                    writer.write(link)


if __name__ == '__main__':
    args = parse_args()
    main(args)
