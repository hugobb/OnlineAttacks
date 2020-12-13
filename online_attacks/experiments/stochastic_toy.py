import torch
import ipdb
import sys
import wandb
import os
import json
import numpy as np
from torch.utils.data import Dataset

sys.path.append("..")  # Adds higher directory to python modules path.
from online_attacks.online_algorithms import create_online_algorithm, compute_knapsack_online_value, compute_competitive_ratio, AlgorithmType, OnlineParams, compute_indices
from online_attacks.utils.utils import seed_everything
from online_attacks.datastream import ToyDatastream, ToyDatastream_Stochastic
from online_attacks.utils.parser import ArgumentParser

def run_experiment(params: OnlineParams, train_loader: Dataset, knapsack:bool):
    offline_algorithm, online_algorithm = create_online_algorithm(params)
    num_perms = len(train_loader)
    comp_ratio_list, online_knapsack_list = [], []
    for i, dataset in enumerate(train_loader):
        offline_dataset, online_dataset = dataset[0], dataset[1]
        online_indices = compute_indices(online_dataset, [online_algorithm])[0]
        offline_indices = compute_indices(offline_dataset, [offline_algorithm])[0]
        comp_ratio_list.append(compute_competitive_ratio(online_indices, offline_indices))
        if knapsack:
            offline_value = sum([x[0] for x in offline_indices])
            online_knapsack_list.append(compute_knapsack_online_value(online_indices))

    # Indicator Competitive Ratio
    comp_ratio = np.sum(comp_ratio_list) / (params.K*num_perms)
    print("Competitive Ratio for %s with K = %d is %f " %(params.online_type, params.K, comp_ratio))

    # Knapsack Competitive Ratio
    if knapsack:
        comp_ratio = np.sum(online_knapsack_list) / (offline_value*num_perms)
        print("Knapsack Competitive Ratio for %s with K = %d is %f " %(params.online_type, params.K, comp_ratio))

    return comp_ratio


def main():
    parser = ArgumentParser(description='Online Attacks')
    # Online params
    parser.add_config("online_params", OnlineParams)

    # Hparams
    parser.add_argument('--K', type=int, default=1, metavar='K',
                        help='Number of attacks to submit')
    parser.add_argument('--eps', type=float, default=1.0, metavar='E',
                        help='Std for noise')
    parser.add_argument('--max_perms', type=int, default=120, metavar='P',
                        help='Maximum number of perms of the data stream')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument("--exhaust", action="store_true", default=False,
            help='Exhaust K')
    parser.add_argument("--knapsack", action="store_true", default=False,
            help='Use Knapsack Competitive Ratio')
    # Bells
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--namestr', type=str, default='Online-Attacks', \
            help='additional info in output filename to describe experiments')


    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='Online-Attacks',
                   name='Online-Attack-{}-{}'.format("toy", args.namestr))
    train_loader = ToyDatastream_Stochastic(args.online_params.N, args.max_perms, args.eps)
    for k in range(1, args.K+1):
        args.online_params.K = k
        comp_ratio = run_experiment(args.online_params, train_loader, args.knapsack)
        if args.wandb:
            model_name = "Competitive Ratio " + args.online_params.online_type.value
            if args.knapsack:
                model_name = "Knapsack Competitive Ratio " + args.online_params.online_type.value
            wandb.log({model_name: comp_ratio, "K": k})


if __name__ == '__main__':
    main()
