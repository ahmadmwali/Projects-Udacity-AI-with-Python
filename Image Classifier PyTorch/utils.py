import os
import json
import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.nn import functional as F

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transform  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transform) 
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transform)
test_dataset  = datasets.ImageFolder(test_dir,  transform = test_transform)

#Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 128)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size = 128)