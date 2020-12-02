from advertorch.attacks import GradientSignAttack
from torch.nn import Module
from .params import AttackerParams


def make_fgsm_attacker(classifier: Module, params: AttackerParams = AttackerParams()) -> GradientSignAttack:
    attacker = GradientSignAttack(classifier, **params)
    return attacker

