import argparse
import torch
from omegaconf import OmegaConf
import copy

# TODO: Fix the import to be more clean !
import sys
import os
path = os.path.realpath(os.path.join(os.getcwd(), ".."))
sys.path.append(path)
import online_attacks.classifiers.mnist as mnist
from online_attacks.classifiers.dataset import DatasetType
from online_attacks.classifiers.mnist.models import MnistModel
from online_attacks.launcher import Launcher
from online_attacks.attacks import Attacker


class TrainClassifier(Launcher):
    @classmethod
    def run(cls, args):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if args.dataset == DatasetType.MNIST:
            params = OmegaConf.structured(mnist.TrainingParams)
            params.model_type = args.model_type
            params.train_on_test = args.train_on_test
            params.name = ""
            if args.robust:
                params.attacker = Attacker.PGD_ATTACK
                params.name = "%s_"%params.attacker.name
            params.name += "test_" if params.train_on_test else "train_"
            params.name += str(args.name)

            mnist.train(params, device=device)
        else:
            raise ValueError()

    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType)
        parser.add_argument("--model_type", nargs='+', type=MnistModel, choices=MnistModel)
        parser.add_argument("--train_on_test", action="store_true")
        parser.add_argument("--num_models", default=1, type=int)
        parser.add_argument("--slurm", type=str, default="")
        parser.add_argument("--robust", action="store_true")
        return parser


if __name__ == "__main__":
    launcher = TrainClassifier()
    parser = TrainClassifier.create_argument_parser()
    args = parser.parse_args()

    for model_type in args.model_type:
        for i in range(args.num_models):
            config = copy.deepcopy(args)
            config.model_type = model_type
            config.name = str(i)
            launcher.launch(config)