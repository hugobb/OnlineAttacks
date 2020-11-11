from advertorch.attacks import Attack
from torch.nn import Module
from .utils import AttackerParams
from .utils import Attacker
from .pgd import make_pgd_attacker


def create_attacker(classifier: Module, params: AttackerParams) -> Attack:
    if params.attacker_type == Attacker.PGD_ATTACK:
        attacker = make_pgd_attacker(classifier, params)
    else:
        raise ValueError()
    
    return attacker
    