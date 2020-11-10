import argparse
import torch
from torch.nn import CrossEntropyLoss

import sys
sys.path.insert(0, ".")
from online_algorithms import create_online_algorithm, AlgorithmType, compute_competitive_ratio
from classifiers.mnist import load_mnist_classifier, MnistModel
from attacks import create_attacker, Attacker
from datastream import datastream
from utils.dataset import load_mnist_dataset


def main(args):
    offline_algorithm, online_algorithm = create_online_algorithm(args.online_type, args.N, args.K)
    
    classifier = load_mnist_classifier(args.model_type)
    classifier.to(args.device)
    criterion = CrossEntropyLoss()

    attacker = create_attacker(args.attacker, classifier)
    dataset = load_mnist_dataset(args.data_dir, args.test)
    transform = datastream.Compose([datastream.ToDevice(args.device), datastream.AttackerTransform(attacker),
                                    datastream.ClassifierTransform(classifier), datastream.LossTransform(criterion)])
    data_stream = datastream.Datastream(dataset, transform=transform)

    comp_ratio = compute_competitive_ratio(data_stream, online_algorithm, offline_algorithm)

    return comp_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Online algorithm arguments
    parser.add_argument('--online_type', type=AlgorithmType, default=AlgorithmType.STOCHASTIC_VIRTUAL,
                         choices=AlgorithmType,)
    parser.add_argument('--N', type=int, default=5, metavar='N',
                        help='Size of datastream')
    parser.add_argument('--K', type=int, default=1, metavar='K',
                        help='Number of attacks to submit')
    
    # MNIST dataset arguments
    parser.add_argument('--data_dir', default="./data", )
    parser.add_argument('--test', action="store_true")


    # MNIST classifier arguments
    parser.add_argument('--model_type', type=MnistModel, default=MnistModel.MODEL_A,
                        choices=MnistModel)

    # Attacker arguments
    parser.add_argument('--attacker', type=Attacker, default=Attacker.PGD_ATTACK, choices=Attacker)
    
    args = parser.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    main(args)

