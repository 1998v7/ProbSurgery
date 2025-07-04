import sys
import inspect
import torch
import copy

from torch.utils.data.dataset import random_split

from datasets_.cars import Cars
from datasets_.dtd import DTD
from datasets_.eurosat import EuroSAT, EuroSATVal
from datasets_.gtsrb import GTSRB
from datasets_.mnist import MNIST
from datasets_.resisc45 import RESISC45
from datasets_.svhn import SVHN
from datasets_.sun397 import SUN397
from datasets_.imagenet100 import ImageNet100
from datasets_.cifar100 import CIFAR100
from datasets_.homeoffice import Homeoffice_Art, Homeoffice_Clipart, Homeoffice_Product, Homeoffice_Real_World

registry = {name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed))
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(new_dataset.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(new_dataset.test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    new_dataset.test_loader_shuffle = torch.utils.data.DataLoader(new_dataset.test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    new_dataset.classnames = copy.copy(dataset.classnames)
    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=2, subset_data_ratio=1.0, val_fraction=0.1, max_val_samples=5000):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers, subset_data_ratio)
            dataset = split_train_into_train_val(base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    
    if dataset_name in ['ImageNet100', 'CIFAR100', 'Homeoffice_Art', "Homeoffice_Clipart", "Homeoffice_Product", "Homeoffice_Real_World"]:
        dataset = dataset_class(preprocess, location=location, batch_size=batch_size, num_workers=num_workers)
    else:
        dataset = dataset_class(preprocess, location=location, batch_size=batch_size, num_workers=num_workers, subset_data_ratio=subset_data_ratio)
    return dataset
