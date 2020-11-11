from torch.utils.data import Dataset, DataLoader
from torch import nn
from advertorch.attacks import Attack
import torch


class BatchDataStream:
    def __init__(self, dataset: Dataset, batch_size: int = 1, transform=None):
        self.dataloader = DataLoader(dataset, batch_size=batch_size)
        self.transform = transform

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        x, target = self.iterator.next()
        if self.transform is not None:
            x, target = self.transform(x, target)
        return x

    def __len__(self):
        return len(self.dataloader)


class PermuteDataset(Dataset):
    def __init__(self, dataset: Dataset, permutation):
        self.dataset = dataset
        self.permutation = permutation
        assert len(permutation) == len(dataset)

    def __getitem__(self, index: int):
        index = self.permutation[index]
        data, target = self.dataset[index]
        target = torch.Tensor([target]).long().squeeze()
        
        return data, target

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
        output = self.criterion(data, target)
        assert len(output) == len(data)
        return output, target
