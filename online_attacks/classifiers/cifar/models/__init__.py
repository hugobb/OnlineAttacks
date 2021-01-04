from enum import Enum
import torch
import torch.nn as nn
import os

from .vgg import VGG
from .resnet import ResNet18
from .densenet import DenseNet
from .googlenet import GoogLeNet
from .wide_resnet import Wide_ResNet


class CifarModel(Enum):
    VGG_16 = "VGG16"
    RESNET_18 = "res18"
    DENSE_121 = "dense121"
    GOOGLENET = "googlenet"
    WIDE_RESNET = "wide_resnet"

    def __str__(self):
        return self.value


__cifar_model_dict__ = {CifarModel.VGG_16: VGG, CifarModel.RESNET_18: ResNet18, CifarModel.DENSE_121: DenseNet,
                        CifarModel.GOOGLENET: GoogLeNet, CifarModel.WIDE_RESNET: Wide_ResNet}


def make_cifar_model(model: CifarModel) -> nn.Module:
    return __cifar_model_dict__[model]()


def load_cifar_classifier(model_type: CifarModel, name: str = None, model_dir: str = None, device=None, eval=False) -> nn.Module:
    model = make_cifar_model(model_type)
    if name is not None:
        filename = os.path.join(model_dir, "mnist", model_type.value, "%s.pth"%name)
        if os.path.exists(filename):
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        else:
            raise OSError("File not found !")
    
    if eval:
        model.eval()
        
    return model.to(device)