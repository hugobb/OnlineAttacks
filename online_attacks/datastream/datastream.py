from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from torch import nn
from advertorch.attacks import Attack
import torch


class BatchDataStream:
    def __init__(self, dataset: Dataset, batch_size: int = 1, transform=None, permutation=None):
        self.dataset = dataset
        if permutation is not None:
            self.dataset = PermuteDataset(self.dataset, permutation=permutation) 

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.transform = transform
        self.batch_size = batch_size

    def __iter__(self):
        for x, target in self.dataloader:
            if self.transform is not None:
                x, target = self.transform(x, target)
            yield x

    def __len__(self):
        return len(self.dataloader)

    def subset(self, index, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=SubsetRandomSampler(index))
        for x, target in dataloader:
            if self.transform is not None:
                x, target = self.transform(x, target)
            yield x, target


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
    def __init__(self, attacker: Attack, target=False):
        self.attacker = attacker
        self.target = target
    
    def __call__(self, data, target):
        if self.target:
            data = self.attacker.perturb(data, target)
        else:
            data = self.attacker.perturb(data)
        return data, target


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
