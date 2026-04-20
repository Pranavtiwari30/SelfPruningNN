# utils/data_loader.py

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
            std=(0.2470, 0.2435, 0.2616)      # CIFAR-10 channel stds
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True,
        download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False,
        download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader