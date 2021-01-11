from .mnist import load_mnist_classifier, MnistModel, load_mnist_dataset
from .cifar import load_cifar_classifier, CifarModel, load_cifar_dataset
from .dataset import DatasetType
from torch.nn import Module


def load_dataset(dataset: DatasetType, train: bool = False):
    if dataset == DatasetType.MNIST:
        return load_mnist_dataset(train=train)
    elif dataset == DatasetType.CIFAR:
        return load_cifar_dataset(train=train)
    else:
        raise ValueError()


def load_classifier(dataset: DatasetType, model_type, name: str = None, model_dir: str = None, device=None, eval=False) -> Module:
    if dataset == DatasetType.MNIST:
        assert isinstance(model_type, MnistModel)
        return load_mnist_classifier(model_type, name=name, model_dir=model_dir, device=device, eval=eval)
    elif dataset == DatasetType.CIFAR:
        assert isinstance(model_type, CifarModel)
        return load_cifar_classifier(model_type, name=name, model_dir=model_dir, device=device, eval=eval)
    else:
        raise ValueError()