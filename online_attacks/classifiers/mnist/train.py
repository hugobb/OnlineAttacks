import torch.nn as nn
import torch
from typing import Optional
import torch.optim as optim
import os
from .params import MnistTrainingParams
from .dataset import create_mnist_loaders
from .models import load_mnist_classifier
from online_attacks.classifiers.trainer import Trainer


def train_mnist(params: MnistTrainingParams = MnistTrainingParams(), device: Optional[torch.device] = None) -> nn.Module:
    train_loader, test_loader = create_mnist_loaders(params.dataset_params)
    model = load_mnist_classifier(params.model_type)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    if params.train_on_test:
        train_loader, test_loader = test_loader, train_loader
    trainer = Trainer(model, train_loader, test_loader, optimizer, device=device)
    
    filename = None
    if params.save_model:
        name = "test" if params.train_on_test else "train"
        filename = os.path.join(params.save_dir, "mnist", params.model_type.value, "%s_%s.pth"%(name,params.name))
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    for epoch in range(1, params.num_epochs):
        trainer.train(epoch)
        trainer.test(epoch)
        if params.save_model:
            torch.save(model.state_dict(), filename)

    return model