from argparse import Namespace
from torch.nn import Module
from mnist import load_mnist_classifier


def load_classifier(args: Namespace, index: int) -> Module:
    filename = 
    if args.dataset == "mnist":
        classifier = load_mnist_classifier(args, index)
    else:
        raise ValueError()

    return classifier