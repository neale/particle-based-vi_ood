import torch
import torchvision
from torchvision import datasets, transforms

import os
import tarfile
import imageio
import tqdm
import numpy as np


def get_classes(target, labels):
    label_indices = []
    for i in range(len(target)):
        if target[i][1] in labels:
            label_indices.append(i)
    return label_indices


def load_mnist(split=False):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    path = 'data_m/'
    if split:
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8, 9]

    else:
        train_idx = list(range(10))
        test_idx = list(range(10))
    
    trainset = datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, train_idx))
    train_loader = torch.utils.data.DataLoader(
            train_hidden,
            batch_size=50,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False)

    valset = datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    val_hidden = torch.utils.data.Subset(valset, get_classes(valset, train_idx))
    val_loader = torch.utils.data.DataLoader(
            val_hidden,
            batch_size=100,
            shuffle=True,
            **kwargs)

    testset = datasets.MNIST(path,
            train=False,
            transform=transforms.Compose([
                #transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
    test_loader = torch.utils.data.DataLoader(
            test_hidden,
            batch_size=100,
            shuffle=False,
            **kwargs)

    return train_loader, test_loader, val_loader


def load_notmnist():
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_nm/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=100, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_fashion_mnist():
    path = 'data_f'
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_cifar10(split=False):
    path = 'data_c/'
    kwargs = {'num_workers': 2}
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  

    if split:
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8, 9]

    else:
        train_idx = list(range(10))
        test_idx = list(range(10))
    
    trainset = torchvision.datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transform)
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, train_idx))
    trainloader = torch.utils.data.DataLoader(
            train_hidden,
            batch_size=128,
            shuffle=True,
            **kwargs)
    
    valset = torchvision.datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transform)
    val_hidden = torch.utils.data.Subset(valset, get_classes(valset, train_idx))
    valloader = torch.utils.data.DataLoader(
            val_hidden,
            batch_size=128,
            shuffle=True,
            **kwargs)

    testset = torchvision.datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transform)
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
    testloader = torch.utils.data.DataLoader(
            test_hidden,
            batch_size=128,
            shuffle=False,
            **kwargs)
    return trainloader, testloader, valloader


def load_cifar100():
    path = 'data_c/'
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    trainset = torchvision.datasets.CIFAR100(root=path, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
            shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root=path, train=False,
            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_svhn(split=False):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': False}
    path = 'data_svhn/'
    if split:
        train_idx = [0, 1, 2, 3, 4, 5]
        test_idx = [6, 7, 8, 9]

    else:
        train_idx = list(range(10))
        test_idx = list(range(10))
    
    trainset = datasets.SVHN(
            path,
            split='train',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ]))
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, train_idx))
    train_loader = torch.utils.data.DataLoader(
            train_hidden,
            batch_size=100,
            shuffle=True,
            **kwargs)
    valset = datasets.SVHN(
            path,
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ]))
    val_hidden = torch.utils.data.Subset(valset, get_classes(valset, train_idx))
    val_loader = torch.utils.data.DataLoader(
            val_hidden,
            batch_size=100,
            shuffle=True,
            **kwargs)

    testset = datasets.SVHN(path,
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ]))

    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
    test_loader = torch.utils.data.DataLoader(
            test_hidden,
            batch_size=100,
            shuffle=False,
            **kwargs)
    return train_loader, test_loader, val_loader



def load_cifarx(c10_idx=[0,1,2,3], c100=False):
    path = './data_c'
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  

    c10_train = c10_idx
    n_cls = len(c10_train)
    
    if c100 == True:
        ranges = list(range(10)) + list(range(40, 50)) + list(range(60, 70)) + list(range(75, 85))
        test_idx = np.random.choice(ranges, size=(10,))

        testset = torchvision.datasets.CIFAR100(root=path, train=False,
                download=True, transform=transform_test)
        test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
        testloader = torch.utils.data.DataLoader(test_hidden, batch_size=100,
                shuffle=True, **kwargs)
    else:
        test_idx = c10_idx
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                download=True, transform=transform_test)
        test_hidden = torch.utils.data.Subset(testset, get_classes(testset, test_idx))
        test_hidden.dataset.targets = torch.tensor(test_hidden.dataset.targets)
        for i, label in enumerate(c10_idx):
            if label > (n_cls - 1):
                test_hidden.dataset.targets[test_hidden.dataset.targets==label] = i
        testloader = torch.utils.data.DataLoader(test_hidden, batch_size=100,
                shuffle=True, **kwargs)

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_train)
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, c10_train))
    train_hidden.dataset.targets = torch.tensor(train_hidden.dataset.targets)
    for i, label in enumerate(c10_idx):
        if label > (n_cls - 1):
            train_hidden.dataset.targets[train_hidden.dataset.targets==label] = i
    trainloader = torch.utils.data.DataLoader(train_hidden, batch_size=100,
            shuffle=True, **kwargs)


    
    return trainloader, testloader

