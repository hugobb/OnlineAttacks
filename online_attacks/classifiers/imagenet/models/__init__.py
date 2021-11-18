from enum import Enum
import torch
import torch.nn as nn
import os
import torchvision.models as models
import ipdb
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



class ImagenetModel(Enum):
    VGG_16 = "VGG16"
    RESNET_18 = "res18"
    DENSE_121 = "dense121"
    GOOGLENET = "googlenet"
    WIDE_RESNET = "wide_resnet"
    MADRY_MODEL = "madry"

    def __str__(self):
        return self.value


__imagenet_model_dict__ = {
    ImagenetModel.VGG_16: models.vgg16(pretrained=True),
    ImagenetModel.RESNET_18: models.resnet18(pretrained=True),
    ImagenetModel.DENSE_121: models.densenet121(pretrained=True),
    ImagenetModel.GOOGLENET: models.googlenet(pretrained=True),
    ImagenetModel.WIDE_RESNET: models.wide_resnet50_2(pretrained=True),
}


def make_imagenet_model(model: ImagenetModel) -> nn.Module:
    return __imagenet_model_dict__[model]


def load_imagenet_classifier(
    model_type: ImagenetModel,
    name: str = None,
    model_dir: str = None,
    device=None,
    eval=False,
) -> nn.Module:
    if model_type == ImagenetModel.MADRY_MODEL:
        from online_attacks.classifiers.madry import load_madry_model

        filename = os.path.join(model_dir, "imagenet", model_type.value, "%s" % name)
        if os.path.exists(filename):
            model = load_madry_model("imagenet", filename)
        else:
            raise OSError("File %s not found !" % filename)

    elif model_type in __imagenet_model_dict__:
        model = make_imagenet_model(model_type)
        # if name is not None:
            # filename = os.path.join(
                # model_dir, "imagenet", model_type.value, "%s.pth" % name
            # )
            # if os.path.exists(filename):
                # state_dict = torch.load(filename, map_location=torch.device("cpu"))
                # model.load_state_dict(state_dict)
            # else:
                # raise OSError("File not found !")

    else:
        raise ValueError()

    if eval:
        model.eval()

    # Hack to be able to use some attacker class
    model.num_classes = 1000

    return model.to(device)
