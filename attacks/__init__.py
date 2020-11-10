from advertorch.attacks import PGDAttack
from enum import Enum


class Attacker(Enum):
    PGD_ATTACK = "pgd"


__attacker_dict__ = {Attacker.PGD_ATTACK: PGDAttack}


def create_attacker(attacker: Attacker, classifier):
    return __attacker_dict__[attacker](classifier)
    