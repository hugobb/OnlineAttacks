from dataclasses import dataclass


@dataclass
class AttackerParams:
    eps: float = 0.3
    clip_min: float = 0.0
    clip_max: float = 1.0
    targeted: bool = False
    # TODO: Add loss_fn
