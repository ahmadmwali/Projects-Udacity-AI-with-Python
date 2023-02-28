# Imports here
import os
import json
import argparse
from time import time

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from collections import OrderedDict
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.nn import functional as F

#Creating an argparse object
parser = argparse.ArgumentParser()

parser.add_argument('data_directory', type = str, 
                    help = 'Data directory')
                    
parser.add_argument('--checkpoint_directory', type = str, default = os.path.join(os.getcwd(),'chekpoint.pth'), 
                    help = 'Filepath to store and load checkpoint')

parser.add_argument('--model', type = str, default = 'vgg11', 
                    help = 'Model architecture from wither of VGG11 or Alexnet')

parser.add_argument('--hidden_units', type = int, default = 512,
                    help = 'Number of nodes in the hidden layer')

parser.add_argument('--epochs', type = int, default = 5,
                    help = 'number of training iterations')
                    
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'Learning rate for GD optimization')
                    
args_in = parser.parse_args()
                    
                    
data_dir = args_in.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
 
#################################################################################################
print('Transforming and loading the data\n')
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
 
print("Finished transforming and loading data\n")
########################################################################################################################
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
                    
#model = models.vgg11(pretrained=True)
#print(model)
                    
#Build and train your network
#model = models.vgg11(pretrained=True)
#print(model)

if args_in.model == 'vgg11':
    model = models.vgg11(pretrained=True)
    features = 25088
    
elif args_in.model == 'alexnet':
    model = models.alexnet(pretrained=True)
    features = 9216
    
else:
    raise ValueError('Model type not available.')
    
for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
nodes = args_in.hidden_units

model.classifier = nn.Sequential(
              nn.Linear(features, nodes),
              nn.ReLU(),
              nn.Dropout(p=0.3),
              nn.Linear(nodes, 256),
              nn.ReLU(),
              nn.Dropout(p=0.3),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args_in.learning_rate)

model.to(device);
##########################################################################
print('Beginning network training\n')

begin = time()

# Network Training

epochs = args_in.epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss  = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    val_loss = criterion(logps, labels)
                    test_loss += val_loss
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()
end = time()
print('Finished network training in ' + str(end-begin) + 's')

########################################## validation on the test set

print('\nBeginning model testing')

model.to(device);

accuracy = 0
model.eval()

with torch.no_grad():
    for inputs, labels in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(test_loader):.3f}")

model.train();
print('Model testing finished\n')
##############################                    
# Save the checkpoint 

print('Saving model chekpoint')
model.class_to_idx = train_dataset.class_to_idx
model.to(device)
checkpoint = {'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict()}
torch.save(checkpoint, args_in.checkpoint_directory)
                    
print('Run finished')    
                    
## Usage
## python train.py flowers --model alexnet
## python train.py flowers