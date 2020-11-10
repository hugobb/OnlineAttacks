from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from argparse import Namespace


def create_mnist_loaders(args: Namespace) -> (DataLoader, DataLoader):
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = MNIST(root=args.data_dir, train=True, transform=transform, download=True)
    testset = MNIST(root=args.data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, num_workers=4)

    return train_loader, test_loader

