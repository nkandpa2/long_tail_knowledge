import os
import subprocess
import shlex
import sys
import argparse

from link_pretraining_data import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--dataset', default='openwebtext', choices=['openwebtext','the_pile','roots','wikipedia'], help='dataset to entity link')
    parser.add_argument('--wikipedia_path', default=None, help='path to wikipedia HFDataset if --dataset wikipedia')
    parser.add_argument('--spotlight_port', nargs='+', type=int, default=[2222], help='ports serving DBPedia Spotlight')
    parser.add_argument('--confidence', default=0.4, type=float, help='DBPedia confidence parameter')
    parser.add_argument('--support', default=20, type=int, help='DBPedia support parameter')
    parser.add_argument('--auth_token', default=None, type=str, help='Huggingface auth token')
    parser.add_argument('--nprocs', default=4, type=int, help='number of processes')
    parser.add_argument('--chunk_size', default=1_000_000, type=int, help='split dataset into chunks of this size')
    args = parser.parse_args()
    return args


def main(args):
    print(f'Loading dataset {args.dataset} to get total number of records')
    dataset = load_dataset(args.dataset, args.wikipedia_path, 0, -1, args.auth_token)
    rows = dataset.nrows
    
    log_dir = os.path.join(args.outdir, 'logs')
    print(f'Setting up output directory {os.path.abspath(args.outdir)}')
    os.makedirs(args.outdir, exist_ok=True)

    print(f'Splitting {rows} rows into chunks of size {args.chunk_size}')
    for start in range(0, rows, args.chunk_size):
        end = min(start + chunk_size, rows)

        dataset_arg = f'--dataset {args.dataset}' 
        wikipedia_arg = f'--wikipedia_path {args.wikipedia_path}' if args.wikipedia_path else ''
        nprocs_arg = f'--nprocs {args.nprocs}'
        spotlight_port_arg = f'--spotlight_port {" ".join([str(p) for p in args.spotlight_port])}'
        confidence_arg = f'--confidence {args.confidence}'
        support_arg = f'--support {args.support}'
        auth_token_arg = f'--auth_token {args.auth_token}'
        
        command f'python link_pretraining_data.py {args.outdir} {dataset_arg} {wikipedia_arg} {nprocs_arg} {spotlight_port_arg} {confidence_arg} {support_arg} {auth_token_arg} --start {start} --end {end}'
        print(f'Processing records {start} to {end}')
        print(command)
        ret = subprocess.Popen(shlex.split(command), stdout=sys.stdout, stderr=sys.stderr).wait()
        print(f'Returned with {ret}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
