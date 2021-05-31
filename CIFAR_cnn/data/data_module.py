# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:38:11 2021

@author: HP
"""
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datasets import DummyDataset

class CIFAR10DataModule(pl.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
      
    # prepare transforms standard to CIFAR10
    self.cifar_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    self.cifar_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
    return self.cifar_train, self.cifar_test
  
  def train_dataloader(self):
    return DataLoader(self.cifar_train, batch_size=64)

  def val_dataloader(self):
    return DataLoader(self.cifar_test, batch_size=64)

data_module = CIFAR10DataModule()
