from advertorch.attacks import PGDAttack
from dataclasses import dataclass
from torch.nn import Module
from .params import AttackerParams


@dataclass
class PGDParams(AttackerParams):
    nb_iter: int = 40
    eps_iter: float = 0.01
    rand_init: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0
    targeted: bool = False


def make_pgd_attacker(classifier: Module, params: PGDParams = PGDParams()) -> PGDAttack:
    
    attacker = PGDAttack(classifier, eps=params.eps, nb_iter=params.nb_iter,
                         eps_iter=params.eps_iter, rand_init=params.rand_init,
                         clip_min=params.clip_min, clip_max=params.clip_max,
                         targeted=params.targeted)

    return attacker

