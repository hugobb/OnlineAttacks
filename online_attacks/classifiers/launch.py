import argparse
import torch
from omegaconf import OmegaConf
import copy

# TODO: Fix the import to be more clean !
import sys
import os
path = os.path.realpath(os.path.join(os.getcwd(), "../.."))
sys.path.append(path)
import online_attacks.classifiers.mnist as mnist
from online_attacks.classifiers.utils.dataset import DatasetType
from online_attacks.classifiers.mnist.models import MnistModel
from online_attacks.launcher import Launcher


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType)
    parser.add_argument("--model_type", nargs='+', type=MnistModel, choices=MnistModel)
    parser.add_argument("--num_models", default=1, type=int)
    parser.add_argument("--slurm", type=str, default="")
    return parser


def run(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset == DatasetType.MNIST:
        params = OmegaConf.structured(mnist.TrainingParams)
        params.model_type = args.model_type
        params.name = args.name
        mnist.train(params, device=device)
    else:
        raise ValueError()


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    launcher = Launcher(run)

    for model_type in args.model_type:
        for i in range(args.num_models):
            config = copy.deepcopy(args)
            config.model_type = model_type
            config.name = str(i)
            launcher.launch(config)