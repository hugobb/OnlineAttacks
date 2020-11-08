import torch
from tqdm import tqdm
from torch import optim
import ipdb
import sys
import wandb
import torch
from torch import autograd
import os
import json
import argparse
import numpy as np

sys.path.append("..")  # Adds higher directory to python modules path.
from online_algorithms import create_online_algorithm

def toy_dataset(N, max_perms=120):
    perms = []
    total_perms = np.math.factorial(N)
    max_perms = np.minimum(max_perms, total_perms)
    for i in range(max_perms):                        # (1) Draw N samples from permutations Universe U (#U = k!)
        while True:                             # (2) Endless loop
            perm = np.random.permutation(N)     # (3) Generate a random permutation form U
            key = tuple(perm)
            if key not in perms:                # (4) Check if permutation already has been drawn (hash table)
                perms.append(key)               # (5) Insert into set
                break
            pass
    train_loader = perms
    return train_loader

def run_experiment(args, K, train_loader):
    offline_algorithm, online_algorithm = create_online_algorithm(args, args.online_type, args.N, K)
    num_perms = len(train_loader)
    comp_ratio = 0.0
    online_indices, offline_indices = None, None

    for i, perm in enumerate(train_loader):
        offline_algorithm.reset()
        online_algorithm.reset()
        for index in range(0, len(perm)):
            online_algorithm.action(perm[index], index)
            offline_algorithm.action(perm[index], index)
        online_indices = set([x[1] for x in online_algorithm.S])
        offline_indices = set([x[1] for x in offline_algorithm.S])
        comp_ratio += len(list(online_indices & offline_indices))
    comp_ratio = comp_ratio / (K*num_perms)
    print("Competitive Ratio for %s with K = %d is %f " %(args.online_type, K, comp_ratio))
    return comp_ratio

def main():
    parser = argparse.ArgumentParser(description='Online Attacks')
    # Hparams
    parser.add_argument('--K', type=int, default=1, metavar='K',
                        help='Number of attacks to submit')
    parser.add_argument('--N', type=int, default=5, metavar='N',
                        help='Size of datastream')
    parser.add_argument('--max_perms', type=int, default=120, metavar='P',
                        help='Maximum number of perms of the data stream')
    # Training
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--online_type', type=str, default='stochastic_virtual')
    # Bells
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')

    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='Online-Attacks',
                   name='Online-Attack-{}-{}'.format(args.dataset, args.namestr))

    for k in range(1, args.K+1):
        train_loader = toy_dataset(args.N, args.max_perms)
        comp_ratio = run_experiment(args, k, train_loader)
        if args.wandb:
            model_name = "Competitive Ratio " + args.online_type
            wandb.log({model_name: comp_ratio, "K": k})

if __name__ == '__main__':
    main()
