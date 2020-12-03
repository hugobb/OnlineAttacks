from dataclasses import dataclass
from argparse import Namespace
from omegaconf import OmegaConf, MISSING
from .models import MnistModel
from online_attacks.classifiers.dataset import DatasetParams
from online_attacks.attacks import Attacker, AttackerParams


@dataclass
class MnistTrainingParams:
    name: str = "mnist"
    model_type: MnistModel = MISSING
    num_epochs: int = 100
    lr: float = 1e-3
    dataset_params: DatasetParams = DatasetParams()
    save_model: bool = True
    save_dir: str = "./pretained_models"
    train_on_test: bool = False
    attacker: Attacker = Attacker.NONE
    attacker_params: AttackerParams = AttackerParams()