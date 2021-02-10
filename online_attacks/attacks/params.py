from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class AttackerParams:
    eps: float = MISSING
    clip_min: float = 0.0
    clip_max: float = 1.0
    targeted: bool = False
    # TODO: Add loss_fn
