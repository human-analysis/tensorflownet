# dataloader.py

import tensorflow as tf
import datasets
import glob
from tqdm import tqdm

class Dataloader:
    """docstring for Dataloader"""
    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.args = args

        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train
        self.dataroot = args.dataroot
        self.batch_size = args.batch_size

        if self.dataset_train_name == "CELEBA":
            self.dataset_train, self.dataset_train_len = datasets.ImageFolder(root=self.dataroot + "/train")

        elif self.dataset_train_name == "MNIST":
            self.dataset_train, self.dataset_train_len = datasets.MNIST(self.dataroot).train()

        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == "CELEBA":
            self.dataset_test, self.dataset_test_len = datasets.ImageFolder(root=self.dataroot + "/test")

        elif self.dataset_test_name == "MNIST":
            self.dataset_test, self.dataset_test_len = datasets.MNIST(self.dataroot).test()

        else:
            raise(Exception("Unknown Dataset"))

    def create(self, shuffle=False, flag=None):
        dataloader = {}
        if flag == "Train":
            dataloader['train'] = (self.dataset_train.batch(self.batch_size).shuffle(self.dataset_train_len), self.dataset_train_len)

        elif flag == "Test":
            dataloader['test'] = (self.dataset_test.batch(self.batch_size), self.dataset_test_len)

        elif flag == None:
            dataloader['train'] = (self.dataset_train.batch(self.batch_size).shuffle(self.dataset_train_len), self.dataset_train_len)
            dataloader['test'] = (self.dataset_test.batch(self.batch_size), self.dataset_test_len)

        return dataloader
