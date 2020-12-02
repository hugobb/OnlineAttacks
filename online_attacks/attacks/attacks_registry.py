from enum import Enum
from advertorch.attacks import Attack
from torch.nn import Module
from .params import AttackerParams


class Attacker(Enum):
    PGD_ATTACK = "pgd"
    FGSM_ATTACK = "fgsm"
    CW_ATTACK = "cw"


def create_attacker(classifier: Module, attacker_type: Attacker,  params: AttackerParams) -> Attack:
    if attacker_type == Attacker.PGD_ATTACK:
        from .pgd import make_pgd_attacker, PGDParams
        params = PGDParams(**params)
        attacker = make_pgd_attacker(classifier, params)
    elif attacker_type == Attacker.FGSM_ATTACK:
        from .fgsm import make_fgsm_attacker
        attacker = make_fgsm_attacker(classifier, params)
    elif attacker_type == Attacker.CW_ATTACK:
        from .cw_pgd import make_cw_attacker, CWParams
        params = CWParams(**params)
        attacker = make_cw_attacker(classifier, params)
    else:
        raise ValueError()
    
    return attacker