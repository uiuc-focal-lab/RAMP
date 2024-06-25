import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
#from utils import download_gdrive
from semidataset import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS
import sys
import pickle

def load_cifar10(n_examples, data_dir='./data', training_set=False, device='cuda'):
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR10(root=data_dir, train=training_set, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_examples].to(device)
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_examples].to(device)
    # x_test = torch.cat([x for (x, y) in test_loader], 0).to(device)
    # y_test = torch.cat([y for (x, y) in test_loader], 0).to(device)

    return x_test, y_test

class BalancedSampler(data.Sampler):
    def __init__(self, labels, batch_size,
                 balanced_fraction=0.5,
                 num_batches=None,
                 label_to_balance=-1, 
                 logger=None):
        logger.info('Inside balanced sampler')
        self.minority_inds = [i for (i, label) in enumerate(labels)
                              if label != label_to_balance]
        self.majority_inds = [i for (i, label) in enumerate(labels)
                              if label == label_to_balance]
        self.batch_size = batch_size
        balanced_batch_size = int(batch_size * balanced_fraction)
        self.minority_batch_size = batch_size - balanced_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.minority_inds) / self.minority_batch_size))

        super().__init__(labels)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            minority_inds_shuffled = [self.minority_inds[i]
                                      for i in
                                      torch.randperm(len(self.minority_inds))]
            # Cycling through permutation of minority indices
            for sup_k in range(0, len(self.minority_inds),
                               self.minority_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = minority_inds_shuffled[
                        sup_k:(sup_k + self.minority_batch_size)]
                # Appending with random majority indices
                if self.minority_batch_size < self.batch_size:
                    batch.extend(
                        [self.majority_inds[i] for i in
                         torch.randint(high=len(self.majority_inds),
                                       size=(self.batch_size - len(batch),),
                                       dtype=torch.int64)])
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

def load_cifar10_train_aug(args, only_train=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor()])

    trainset = SemiSupervisedDataset(base_dataset='cifar10',
                                 root=args.data_dir, train=True,
                                 download=True,
                                 transform=transform_train,
                                 aux_data_filename='ti_500K_pseudo_labeled.pickle')

    # num_batches=50000 enforces the definition of an "epoch" as passing through 50K
    # datapoints
    train_batch_sampler = SemiSupervisedSampler(
        trainset.sup_indices, trainset.unsup_indices,
        args.batch_size, 0.5,
        num_batches=int(np.ceil(100000 / args.batch_size)))
    # epoch_size = len(train_batch_sampler) * args.batch_size

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler)

    testset = SemiSupervisedDataset(base_dataset='cifar10',
                                    root=args.data_dir, train=False,
                                    download=True,
                                    transform=transform_test)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False)

    return train_loader, test_loader



def load_imagenet(n_examples, training_set=False, return_loader=False, device='cpu'):
    IMAGENET_SL = 224
    if not training_set:
        IMAGENET_PATH = "/share/datasets/imagenet/val"
        if not os.path.exists(IMAGENET_PATH):
            IMAGENET_PATH = "/share/datasets/imagenet/val_orig"
    else:
        IMAGENET_PATH = "/share/datasets/imagenet/train"
    imagenet = datasets.ImageFolder(IMAGENET_PATH,
                           transforms.Compose([
                               transforms.Resize(IMAGENET_SL + 32),
                               transforms.CenterCrop(IMAGENET_SL),
                               transforms.ToTensor()
                           ]))
    torch.manual_seed(0)

    # transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    # ])

    test_loader = data.DataLoader(imagenet, batch_size=n_examples, shuffle=True, num_workers=30)
    
    if return_loader:
        from robustness.tools import helpers
        return helpers.DataPrefetcher(test_loader)
    testiter = iter(test_loader)
    x_test, y_test = next(testiter)

    # x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_examples].to(device)
    # y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_examples].to(device)
    
    return x_test.to(device), y_test.to(device)

#

def load_anydataset(args, device='cuda'):
    if args.dataset == 'cifar10':
        x_test, y_test = load_cifar10(args.n_ex, args.data_dir,
            args.training_set, device=device)
        #x_test = x_test.contiguous()
    elif args.dataset == 'imagenet':
        x_test, y_test = load_imagenet(args.n_ex, args.training_set)
    elif args.dataset == 'cifar100':
        x_test, y_test = load_cifar100(args.n_ex, '/home/scratch/datasets/CIFAR100',
            args.training_set, device=device)
    elif args.dataset == 'imagenet100':
        x_test, y_test = load_imaget100(args.n_ex)
    
    return x_test, y_test

def load_cifar100(n_examples, data_dir='/home/scratch/datasets/CIFAR100', training_set=False, device='cuda'):
    transform_chain = transforms.Compose([transforms.ToTensor()])
    item = datasets.CIFAR100(root=data_dir, train=training_set, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_examples].to(device)
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_examples].to(device)

    return x_test, y_test

def load_cifar10_train(args, only_train=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    root = args.data_dir + '' #'/home/EnResNet/WideResNet34-10/data/'
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        root, train=True, transform=train_transform, download=True)
    if not only_train:
      test_dataset = datasets.CIFAR10(
          root, train=False, transform=test_transform, download=True)
      
    if args.n_examples != 0:
        subset_indices = [i for i in range(args.n_examples)]  # Replace 5000 with the size of the subset you want

        # Create a subset of the CIFAR-10 dataset
        subset_dataset = Subset(train_dataset, subset_indices)

        # Create a data loader for the subset
        train_loader = DataLoader(subset_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_workers,)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
    if not only_train:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_eval,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
    else:
        test_loader = ()
    
    return train_loader, test_loader
    
    
# data loaders training
# def load_cifar10_train(args, only_train=False):
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         #transforms.RandomRotation(15),
#         transforms.ToTensor(),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     root = args.data_dir + '' #'/home/EnResNet/WideResNet34-10/data/'
#     num_workers = 2
#     train_dataset = datasets.CIFAR10(
#         root, train=True, transform=train_transform, download=True)
#     if not only_train:
#       test_dataset = datasets.CIFAR10(
#           root, train=False, transform=test_transform, download=True)
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=num_workers,
#     )
#     if not only_train:
#         test_loader = torch.utils.data.DataLoader(
#             dataset=test_dataset,
#             batch_size=args.batch_size_eval,
#             shuffle=False,
#             pin_memory=True,
#             num_workers=0,
#         )
#     else:
#         test_loader = ()
    
#     return train_loader, test_loader

def load_imagenet_train(args):
    from robustness.datasets import DATASETS
    from robustness.tools import helpers
    dataset = DATASETS['imagenet'](args.data_dir) #'/home/scratch/datasets/imagenet'
    
    
    train_loader, val_loader = dataset.make_loaders(30,
                    args.batch_size, data_aug=True)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    return train_loader, val_loader

if __name__ == '__main__':
    #x_test, y_test = load_cifar10c(100, corruptions=['fog'])
    x_test, y_test = load_imagenet100(100)
    print(x_test.shape, x_test.max(), x_test.min())


