# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:51:52 2021

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


def print_accuracy(top_class, labels):
  if top_class.shape != labels.shape:
    labels.view(*top_class.shape)
  equal = top_class == labels
  accuracy = torch.mean(equal.type(torch.FloatTensor))
  print(f'Accuracy: {accuracy.item()*100}%')
  
  
def calculate_accuracy(top_class, labels):
  if top_class.shape != labels.shape:
    labels.view(*top_class.shape)
  equal = top_class == labels
  accuracy = torch.mean(equal.type(torch.FloatTensor))
  return accuracy.item()

def train_batch(model, images, labels, loss_func, optimizer = None):
  if torch.cuda.is_available()==True:
    images = images.type(torch.float).to(torch.device("cuda:0"))
    labels = labels.to(torch.device("cuda:0"))

  label_pred = model(images)

  loss = loss_func(label_pred, labels)
  top_prob, top_class = torch.exp(label_pred).topk(1, dim = 1)
  batch_accuracy = calculate_accuracy(top_class, labels)

  if optimizer is not None:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return loss.item(), batch_accuracy


def train_epoch(model, trainloader, testloader, optimizer, epoch, loss_func):
  running_loss = 0.
  test_loss = 0.
  train_accuracy = 0.
  val_accuracy = 0.

  model.train()
  for im, label in trainloader:
    optimizer.zero_grad()
    x, y = train_batch(model, im, label, loss_func, optimizer)
    running_loss += x
    train_accuracy += y

  with torch.no_grad():
    model.eval()
    for im_test, label_test in testloader:
      x, y = train_batch(model, im_test, label_test, loss_func)
      test_loss += x
      val_accuracy += y
    
    model.train()
  return running_loss/len(trainloader), train_accuracy/len(trainloader), test_loss/len(testloader), val_accuracy/len(testloader)



from torch.utils.tensorboard import SummaryWriter
import numpy as np

def visualiser_tensorboard(trainloss_list, testloss_list, trainacc_list, valacc_list):
  writer = SummaryWriter()
  for n_iter in range(len(trainloss_list)):
    writer.add_scalar('Loss/train', np.array(trainloss_list).squeeze()[n_iter], n_iter)
    writer.add_scalar('Loss/test', np.array(testloss_list).squeeze()[n_iter], n_iter)
    writer.add_scalar('Accuracy/train', np.array(trainacc_list).squeeze()[n_iter], n_iter)
    writer.add_scalar('Accuracy/test', np.array(valacc_list).squeeze()[n_iter], n_iter)
  writer.close()
  
  
def train_model(epochs, model, optimizer, trainloader, testloader, loss_func):
  train_losses = []
  test_losses = []
  train_accuracies = []
  val_accuracies = []
  
  for epoch in range(epochs):
    x, y, z, t = train_epoch(model, trainloader, testloader, optimizer, epoch, loss_func)
    train_losses.append(x)
    test_losses.append(y)
    train_accuracies.append(z)
    val_accuracies.append(t)

  print("Epoch: {}/{}..".format(epoch, epochs),
      "Training Loss: {:.3f}..".format(train_losses[-1]),
      "Test Loss: {:.3f}..".format(test_losses[-1]),
      "Test Accuracy: {:.3f}..".format(val_accuracies[-1]),
      "Training Accuracy: {:.3f}..".format(train_accuracies[-1]))
  visualiser_tensorboard(train_losses, test_losses, train_accuracies, val_accuracies )



