from torch import nn
import torch.nn.functional as F
import torch
from torch import optim
from enum import Enum
from argparse import Namespace
import os


import sys
sys.path.insert(0, ".")  # Adds higher directory to python modules path.
from utils.dataset import create_mnist_loaders
from classifiers.train import Trainer


class modelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class modelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.conv2 = nn.Conv2d(64, 128, 6)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 12 * 12, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class modelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class modelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 300)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 300)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(300, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 300)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x


class MnistModel(Enum):
    MODEL_A = "modelA"
    MODEL_B = "modelB"
    MODEL_C = "modelC"
    MODEL_D = "modelD"


__mnist_model_dict__ = {MnistModel.MODEL_A: modelA, MnistModel.MODEL_B: modelB, MnistModel.MODEL_C: modelC, MnistModel.MODEL_D: modelD}


def train_mnist_classifier(model_type: MnistModel, filename: str, args: Namespace, num_epochs: int = 100, device=None) -> nn.Module:
    train_loader, test_loader = create_mnist_loaders(args.data_dir, args)
    model = load_mnist_classifier(model_type)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, train_loader, test_loader, optimizer, device=device)
    
    for epoch in range(1, num_epochs):
        trainer.train(epoch)
        trainer.test(epoch)

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    torch.save(model.state_dict(), filename)
    return model


def load_mnist_classifier(model_type: MnistModel, index: int = None, model_dir: str = None) -> nn.Module:
    model = __mnist_model_dict__[model_type]()
    if index is None:
        return model

    filename = os.path.join(model_dir, "mnist", model_type, "%i.pth"%index)
    if os.path.exists(filename):
        state_dict = torch.load(filename)
        model.load_state_dict(state_dict)
    else:
        raise OSError("File not found !")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MnistModel.MODEL_A, type=MnistModel, choices=MnistModel)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--filename', type=str, default="./pretrained_models/mnist/model.pth")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--data_dir', type=str, default='./data')

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(None)

    args = parser.parse_args()
    train_mnist_classifier(args.model, args.filename, args, args.num_epochs, device=device)