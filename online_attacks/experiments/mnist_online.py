
import torch
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from online_attacks.online_algorithms import create_online_algorithm, compute_competitive_ratio, OnlineParams, AlgorithmType, compute_indices
from online_attacks.classifiers.mnist import load_mnist_classifier, MnistModel, load_mnist_dataset
from online_attacks.attacks import create_attacker, Attacker, AttackerParams
from online_attacks.datastream import datastream
from online_attacks.utils.parser import ArgumentParser
import numpy as np


@dataclass
class MnistParams:
    attacker_type: Attacker = Attacker.PGD_ATTACK
    attacker_params: AttackerParams = AttackerParams()
    online_params: OnlineParams = OnlineParams(K=100, online_type=AlgorithmType.STOCHASTIC_OPTIMISTIC)
    model_dir: str = "/checkpoint/hberard/OnlineAttack/pretained_models/"


def run(args, params: MnistParams = MnistParams()):
    dataset = load_mnist_dataset(train=False)
    dataset = datastream.PermuteDataset(dataset, permutation=np.random.permutation(len(dataset)))

    target_classifier = load_mnist_classifier(args.model_type, index=0, model_dir=params.model_dir, device=args.device, eval=True)
    source_classifier = load_mnist_classifier(args.model_type, index=1, model_dir=params.model_dir, device=args.device, eval=True)

    criterion = CrossEntropyLoss(reduction="none")

    attacker = create_attacker(source_classifier, params.attacker_type, params.attacker_params)
    
    transform = datastream.Compose([datastream.ToDevice(args.device), datastream.AttackerTransform(attacker),
                                    datastream.ClassifierTransform(target_classifier), datastream.LossTransform(criterion)])
    target_stream = datastream.BatchDataStream(dataset, batch_size=1000, transform=transform)

    transform = datastream.Compose([datastream.ToDevice(args.device), datastream.AttackerTransform(attacker),
                                    datastream.ClassifierTransform(source_classifier), datastream.LossTransform(criterion)])
    source_stream = datastream.BatchDataStream(dataset, batch_size=1000, transform=transform)

    params.online_params.N = len(dataset)
    offline_algorithm, online_algorithm = create_online_algorithm(params.online_params)
    print("Computing indices...")
    source_online_indices, source_offline_indices = compute_indices(source_stream, [online_algorithm, offline_algorithm], pbar_flag=True)
    target_offline_indices = compute_indices(target_stream, [offline_algorithm], pbar_flag=True)[0]

    print("Computing Competitive Ratio...")
    comp_ratio = compute_competitive_ratio(source_online_indices, source_offline_indices)
    print("Comp ratio source online vs target online: %.2f"%(comp_ratio/params.online_params.K))
    comp_ratio = compute_competitive_ratio(source_online_indices, target_offline_indices)
    print("Comp ratio source online vs target offline: %.2f"%(comp_ratio/params.online_params.K))
    comp_ratio = compute_competitive_ratio(source_offline_indices, target_offline_indices)
    print("Comp ratio target online vs target offline: %.2f"%(comp_ratio/params.online_params.K))

    return comp_ratio


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Online algorithm arguments
    parser.add_config("mnist_params", MnistParams)
    
    # MNIST dataset arguments
    parser.add_argument('--data_dir', default="./data", )
    parser.add_argument('--test', action="store_true")


    # MNIST classifier arguments
    parser.add_argument('--model_type', type=MnistModel, default=MnistModel.MODEL_A,
                        choices=MnistModel)

    # Attacker arguments
    parser.add_argument('--attacker', type=Attacker, default=Attacker.PGD_ATTACK, choices=Attacker)
    
    args = parser.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run(args, args.mnist_params)

