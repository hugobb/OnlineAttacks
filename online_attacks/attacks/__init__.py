from .attacks_registry import create_attacker, Attacker, AttackerParams
from typing import Iterable


def compute_attack_success_rate(datastream: Iterable):
    adv_correct = 0
    num_samples = 0
    for x, target in datastream:
        pred = x.max(1, keepdim=True)[1]
        adv_correct += pred.eq(target.view_as(pred)).sum().item()
        num_samples += len(x)
    fool_rate = 1 - adv_correct / num_samples
    
    return fool_rate