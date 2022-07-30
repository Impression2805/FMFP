import os
import numpy as np
import torch
import sys
sys.path.append('../')
import torchvision as tv
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import csv, torchvision, random, os
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            len_train = int(np.floor(0.9 * len(self.dataset)))
            return (len_train+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    def __init__(self, dataset, dataname, indices=None):
        self.dataname = dataname
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices
        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


def get_loader(data, data_path, batch_size, args):
    # dataset normalize values
    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]

    # augmentation
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # load datasets
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,transform=train_transforms,download=True)
        eval_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,transform=test_transforms,download=False)
        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False, transform=test_transforms, download=False)

    elif data == 'cifar10':
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,transform=train_transforms,download=True)
        eval_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,transform=test_transforms,download=False)
        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False, transform=test_transforms, download=False)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    X_train_total = np.array(train_set.data)
    Y_train_total = np.array(train_set.targets)
    X_train = X_train_total[train_idx]
    X_valid = X_train_total[val_idx]
    Y_train = Y_train_total[train_idx]
    Y_valid = Y_train_total[val_idx]
    train_set.data = X_train.astype('uint8')
    train_set.targets = Y_train
    eval_set.data = X_valid.astype('uint8')
    eval_set.targets = Y_valid

    method = None
    train_data = Custom_Dataset(train_set.data,train_set.targets,'cifar', train_transforms, method=method)
    eval_data = Custom_Dataset(eval_set.data, eval_set.targets,'cifar', test_transforms)
    test_data = Custom_Dataset(test_set.data, test_set.targets, 'cifar', test_transforms)
    test_onehot = one_hot_encoding(test_set.targets)
    test_label = test_set.targets

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=4)

    print("-------------------Make loader-------------------")
    print('Train Dataset :',len(train_loader.dataset), 'Valid Dataset :',len(valid_loader.dataset),
          '   Test Dataset :',len(test_loader.dataset))
    return train_loader, valid_loader, test_loader, test_onehot, test_label


class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None, method=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform
        self.method = method

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        if self.method == None:
            x = self.transform(img)
        else:
            x = img
        return x, self.y_data[idx], idx


def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot

