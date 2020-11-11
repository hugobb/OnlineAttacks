from advertorch.attacks import PGDAttack
from dataclasses import dataclass
from torch.nn import Module
from .utils import AttackerParams


@dataclass
class PGDParams:
    nb_iter: int = 40
    eps_iter: float = 0.01
    rand_init: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0
    targeted: bool = False


def make_pgd_attacker(classifier: Module, params: AttackerParams = AttackerParams(),
                      pgd_params: PGDParams = PGDParams()) -> PGDAttack:
    
    attacker = PGDAttack(classifier, eps=params.eps, nb_iter=pgd_params.nb_iter,
                         eps_iter=pgd_params.eps_iter, rand_init=pgd_params.rand_init,
                         clip_min=pgd_params.clip_min, clip_max=pgd_params.clip_max,
                         targeted=pgd_params.targeted)

    return attacker

