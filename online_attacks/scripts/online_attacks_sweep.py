from dataclasses import dataclass
import torch
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf, MISSING
import tqdm
from typing import Any

from online_attacks.classifiers import load_dataset, DatasetType, load_classifier, MnistModel, CifarModel
from online_attacks.attacks import create_attacker, Attacker, AttackerParams
from online_attacks.datastream import datastream
from online_attacks.online_algorithms import create_algorithm, OnlineParams, compute_indices, AlgorithmType
from online_attacks.utils.logger import Logger, config_exists
from online_attacks.utils import seed_everything


@dataclass
class Params:
    name: str = "default"
    dataset: DatasetType = MISSING
    model_name: str = MISSING
    model_type: Any = MISSING
    model_dir: str = "/checkpoint/hberard/OnlineAttack/pretained_models/"
    attacker_type: Attacker = MISSING
    attacker_params: AttackerParams = AttackerParams()
    online_params: OnlineParams = OnlineParams(exhaust=True)
    save_dir: str = "/checkpoint/hberard/OnlineAttack/results_icml/${name}"
    seed: int = 1234
    batch_size: int = 1000


@dataclass
class CifarParams(Params):
    model_type: CifarModel = MISSING


@dataclass
class MnistParams(Params):
    model_type: MnistModel = MISSING


class OnlineAttackExp:
    @staticmethod
    def create_params(params):
        if params.dataset == DatasetType.CIFAR:
            params = OmegaConf.structured(CifarParams(**params))
        if params.dataset == DatasetType.MNIST:
            params = OmegaConf.structured(MnistParams(**params))
        return params

    @staticmethod
    def run(params: Params, num_runs: int = 1):
        seed_everything(params.seed)
        
        exp_id = config_exists(params)
        logger = Logger(params.save_dir, exp_id=exp_id)
        if exp_id is None:
            logger.save_hparams(params)
        
        num_runs -= len(logger) 
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = load_dataset(params.dataset, train=False)
        permutation_gen = datastream.PermutationGenerator(len(dataset), seed=params.seed)

        source_classifier = load_classifier(params.dataset, params.model_type, name=params.model_name, model_dir=params.model_dir, device=device, eval=True)
        attacker = create_attacker(source_classifier, params.attacker_type, params.attacker_params)

        transform = datastream.Compose([datastream.ToDevice(device), datastream.AttackerTransform(attacker),
                                        datastream.ClassifierTransform(source_classifier), datastream.LossTransform(CrossEntropyLoss(reduction="none"))])
        
        algorithm = create_algorithm(params.online_params.online_type, params.online_params, N=len(dataset))

        
        for i in tqdm.tqdm(range(num_runs)):
            permutation = permutation_gen.sample()
            source_stream = datastream.BatchDataStream(dataset, batch_size=params.batch_size, transform=transform, permutation=permutation)
            indices = compute_indices(source_stream, algorithm, pbar_flag=False)
            record = {"permutation": permutation.tolist(), "indices": indices}
            logger.save_record(record)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DatasetType.MNIST, type=DatasetType, choices=DatasetType)
    parser.add_argument("--model_name", default="0", type=str)
    parser.add_argument("--attacker_type", default=Attacker.FGSM_ATTACK, type=Attacker, choices=Attacker)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--name", default="default", type=str)

    
    # Hack to be able to parse either MnistModel or CifarModel 
    args, _ = parser.parse_known_args()
    if args.dataset == DatasetType.MNIST:
        parser.add_argument("--model_type", default=MnistModel.MODEL_A, type=MnistModel, choices=MnistModel)
    elif args.dataset == DatasetType.CIFAR:
        parser.add_argument("--model_type", default=CifarModel.VGG_16, type=CifarModel, choices=CifarModel)
    args, _ = parser.parse_known_args()
    
    params = OmegaConf.structured(Params(**vars(args)))
    params = OnlineAttackExp.create_params(params)

    # Hack to be able to parse num_runs without affecting params
    parser.add_argument("--num_runs", default=100, type=int)
    args = parser.parse_args()
 
    for k in [10, 100, 1000]:
        params.online_params.K = k
        params.online_params.online_type = list(AlgorithmType)
        OnlineAttackExp.run(params, args.num_runs)

