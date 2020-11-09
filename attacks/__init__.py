from advertorch.attacks import PGDAttack, Attack
from torch.nn import Module
from argparse import Namespace


def create_attacker(attacker_type, classifier: Module, args: Namespace) -> Attack:
    if attacker_type == "pgd":
        attacker = PGDAttack(classifier, args)
    else:
        raise ValueError()

    return attacker