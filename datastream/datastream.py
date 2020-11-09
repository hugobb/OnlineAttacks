from torch.utils.data import Dataset
from torch import nn
from advertorch.attacks import Attack


class ClassifierDatastream(Dataset):
    def __init__(self, dataset: Dataset, classifier: nn.Module):
        self.dataset = dataset
        self.classifier = classifier

    def __getitem__(self, index: int):
        data = self.dataset[i]
        data = self.classifier(data)
        return data

    def __len__(self) -> int:
        return len(self.dataset)


class AttackDatastream(Dataset):
    def __init__(self, dataset: Dataset, attacker: Attack):
        self.dataset = dataset
        self.attacker = attacker

    def __getitem__(self, index: int):
        data = self.dataset[i]
        data = self.attacker.perturb(data)
        return data

    def __len__(self) -> int:
        return len(self.dataset)