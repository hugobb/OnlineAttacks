from torch.utils.data import Dataset
from torch import nn
from advertorch.attacks import Attack
import torch


class Datastream(Dataset):
    def __init__(self, dataset: Dataset, transform=None, permutation=None):
        self.dataset = dataset
        self.transform = transform
        self.permutation = permutation
        if permutation is not None:
            assert len(permutation) == len(dataset)

    def __getitem__(self, index: int):
        if self.permutation is not None:
            index = self.permutation[index]
        data, target = self.dataset[index]
        data = data.unsqueeze(0)
        target = torch.Tensor([target]).long()
        
        if self.transform is not None:
            data, target = self.transform(data, target)
        return data

    def __len__(self) -> int:
        return len(self.dataset)


class Compose:
    def __init__(self, list_transforms: list):
        self.list_transforms = list_transforms

    def __call__(self, data, target):
        for transform in self.list_transforms:
            data, target = transform(data, target)
        return data, target


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, data, target):
        return data.to(self.device), target.to(self.device)


class AttackerTransform:
    def __init__(self, attacker: Attack):
        self.attacker = attacker
    
    def __call__(self, data, target):
        return self.attacker.perturb(data, target), target


class ClassifierTransform:
    def __init__(self, classifier: nn.Module):
        self.classifier = classifier
    
    def __call__(self, data, target):
        return self.classifier(data), target


class LossTransform:
    def __init__(self, criterion: nn.Module):
        self.criterion = criterion
    
    def __call__(self, data, target):
        return self.criterion(data, target), target
