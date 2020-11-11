from enum import Enum
from advertorch.attacks import Attack
from torch.nn import Module
from omegaconf import OmegaConf
from .params import AttackerParams
from .pgd import make_pgd_attacker, PGDParams


class Attacker(Enum):
    PGD_ATTACK = "pgd"


def create_attacker(classifier: Module, attacker_type: Attacker,  params: AttackerParams) -> Attack:
    if attacker_type == Attacker.PGD_ATTACK:
        params = OmegaConf.merge(params, OmegaConf.structured(PGDParams()))
        attacker = make_pgd_attacker(classifier, params)
    else:
        raise ValueError()
    
    return attacker