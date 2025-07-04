import os
import torch
import torchvision.datasets as datasets

class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

class ImageNet100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('./data'),
                 batch_size=32,
                 num_workers=16):

        traindir = os.path.join(location, 'ImageNet100', 'train')
        valdir = os.path.join(location, 'ImageNet100', 'val')

        self.train_dataset = ImageFolderDataset(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]