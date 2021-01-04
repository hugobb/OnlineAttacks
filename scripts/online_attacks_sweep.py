from dataclasses import dataclass
import torch
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf

from online_attacks.classifiers.mnist import load_mnist_classifier, load_mnist_dataset, MnistModel
from online_attacks.attacks import create_attacker, Attacker, AttackerParams
from online_attacks.datastream import datastream
from online_attacks.online_algorithms import create_algorithm, OnlineParams, compute_indices, AlgorithmType
from online_attacks.utils.logger import Logger, LoggerParams
from online_attacks.utils import seed_everything


@dataclass
class Params:
    num_runs: int = 5
    model_type: MnistModel = MnistModel.MODEL_A
    model_name: str = "0"
    model_dir: str = "/checkpoint/hberard/OnlineAttack/pretained_models/"
    attacker_type: Attacker = Attacker.FGSM_ATTACK
    attacker_params: AttackerParams = AttackerParams()
    online_params: OnlineParams = OnlineParams(exhaust=False)
    logger_params: LoggerParams = LoggerParams(save_dir="/checkpoint/hberard/OnlineAttack/new_results/fgsm_not_robust_no_exhaust")
    seed: int = 1234


def run(params: Params):
    seed_everything(params.seed)
    
    logger = Logger(params.logger_params)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_mnist_dataset(train=False)
    permutation_gen = datastream.PermutationGenerator(len(dataset), seed=params.seed)

    params.online_params.N = len(dataset)

    source_classifier = load_mnist_classifier(params.model_type, name=params.model_name, model_dir=params.model_dir, device=device, eval=True)
    attacker = create_attacker(source_classifier, params.attacker_type, params.attacker_params)

    transform = datastream.Compose([datastream.ToDevice(device), datastream.AttackerTransform(attacker),
                                    datastream.ClassifierTransform(source_classifier), datastream.LossTransform(CrossEntropyLoss(reduction="none"))])
    
    algorithm = create_algorithm(params.online_params.online_type, params.online_params)

    record = {"hparams": OmegaConf.to_container(params), "runs": []}
    for i in range(params.num_runs):
        permutation = permutation_gen.sample()
        source_stream = datastream.BatchDataStream(dataset, batch_size=1000, transform=transform, permutation=permutation)
        indices = compute_indices(source_stream, algorithm, pbar_flag=False)
        record["runs"].append({"permutation": permutation.tolist(), "indices": indices})
    
    logger.save_record(record)
    logger.save_hparams(params)


if __name__ == "__main__":
    for k in [10, 100, 1000]:
        params = OmegaConf.structured(Params())
        params.online_params.K = k
        params.online_params.online_type = list(AlgorithmType)
        run(params)

