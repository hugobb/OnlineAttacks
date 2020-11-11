from enum import Enum
from dataclasses import dataclass


class Attacker(Enum):
    PGD_ATTACK = "pgd"


@dataclass
class AttackerParams:
    attacker_type: Attacker = Attacker.PGD_ATTACK
    eps: float = 0.3
    # TODO: Add loss_fn
