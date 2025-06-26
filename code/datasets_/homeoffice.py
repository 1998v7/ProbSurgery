import os
import torch
import torchvision.datasets as datasets
import re
import numpy as np

class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)
        self.indices = np.arange(len(self.samples))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
    

class Homeoffice_Art:
    def __init__(self,
                 preprocess,
                 test_split=None,
                 location='./data',
                 batch_size=32,
                 num_workers=16):
        
        domain = ["Art", "Clipart", "Product", "Real World"]
        traindir = os.path.join(location, "OfficeHome_split", "Art", "train")
        valdir = os.path.join(location, "OfficeHome_split","Art", "val")


        self.train_dataset = ImageFolderDataset(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]


class Homeoffice_Clipart:
    def __init__(self,
                 preprocess,
                  test_split=None,
                 location='./data',
                 batch_size=32,
                 num_workers=16):
        
        domain = ["Art", "Clipart", "Product", "Real World"] 
        traindir = os.path.join(location, "OfficeHome_split", "Clipart", 'train')
        valdir = os.path.join(location, "OfficeHome_split","Clipart", 'val')


        self.train_dataset = ImageFolderDataset(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]


class Homeoffice_Product:
    def __init__(self,
                 preprocess,
                 test_split=None,
                 location='./data',
                 batch_size=32,
                 num_workers=16):
        
        domain = ["Art", "Clipart", "Product", "Real World"]
        traindir = os.path.join(location, "OfficeHome_split","Product", 'train')
        valdir = os.path.join(location, "OfficeHome_split", "Product", 'val')


        self.train_dataset = ImageFolderDataset(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]


class Homeoffice_Real_World:
    def __init__(self,
                 preprocess,
                 test_split=None,
                 location='./data',
                 batch_size=32,
                 num_workers=16):
        
        domain = ["Art", "Clipart", "Product", "Real World"]
        traindir = os.path.join(location, "OfficeHome_split", "Real World", 'train')
        valdir = os.path.join(location, "OfficeHome_split", "Real World", 'val')


        self.train_dataset = ImageFolderDataset(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]