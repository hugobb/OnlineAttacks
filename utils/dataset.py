from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from argparse import Namespace


def load_mnist_dataset(data_dir: str, train: bool=True) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    return MNIST(root=data_dir, train=train, transform=transform, download=True)


def create_mnist_loaders(data_dir: str, args: Namespace) -> (DataLoader, DataLoader):
    trainset = load_mnist_dataset(data_dir, True)
    testset = load_mnist_dataset(data_dir, False)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, num_workers=4)

    return train_loader, test_loader

