import torchvision.transforms as transforms
import torchvision
import numpy as np
from torch.utils.data import TensorDataset
import torch


def get_transforms(dataset, data_augmentation):
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return transform, transform  # No data augmentation for mnist, to simple
    elif dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),  # randomly flip the images horizontally
                transforms.RandomCrop(32, padding=4),  # randomly crop the images
                transforms.ToTensor(),  # convert to tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize with mean and std
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize with mean and std
            ]
        )
        if not data_augmentation:
            return transform_test, transform_test
        else:
            return transform_train, transform_test


def load_mnist(train, data_augmentation, seed):
    if not train and data_augmentation:
        raise ValueError("You should not use data augmentation with test data")

    transform_train, transform_test = get_transforms(dataset="mnist", data_augmentation=data_augmentation)
    if train:
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        train_set, val_set = torch.utils.data.random_split(
            trainset, [50000, 10000], generator=torch.Generator().manual_seed(seed)
        )
        return train_set, val_set
    else:
        testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
        return testset


def convert_to_tensordataset(dataset):
    x = np.zeros((len(dataset), 28, 28))
    y = np.zeros((len(dataset)), dtype=np.int32)

    for i in range(len(dataset)):
        x[i, :, :] = dataset[i][0]
        y[i] = dataset[i][1]

    x = x.astype(float)
    y = y.astype(int)

    x = np.expand_dims(x, axis=1)  # add the channel dimension

    return TensorDataset(torch.Tensor(x), torch.LongTensor(y))


def load_fashionmnist(train, data_augmentation, seed):
    if not train and data_augmentation:
        raise ValueError("You should not use data augmentation with test data")

    transform_train, transform_test = get_transforms(dataset="mnist", data_augmentation=data_augmentation)
    if train:
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        train_set, val_set = torch.utils.data.random_split(
            trainset, [50000, 10000], generator=torch.Generator().manual_seed(seed)
        )
        return convert_to_tensordataset(train_set), convert_to_tensordataset(val_set)
    else:
        testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
        return convert_to_tensordataset(testset)


def load_cifar10(train, data_augmentation, seed):
    transform_train, transform_test = get_transforms(dataset="cifar10", data_augmentation=data_augmentation)
    if train:
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        train_set, val_set = torch.utils.data.random_split(
            trainset, [40000, 10000], generator=torch.Generator().manual_seed(seed)
        )
        return train_set, val_set
    else:
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        return testset
