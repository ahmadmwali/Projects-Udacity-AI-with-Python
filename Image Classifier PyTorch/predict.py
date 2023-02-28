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


parser = argparse.ArgumentParser()

parser.add_argument('img_path', type = str, help = 'path to image to predict')

parser.add_argument('--checkpoint_directory', type = str, default = os.path.join(os.getcwd(),'chekpoint.pth'),
                    help = 'Filepath to store and load checkpoint')

parser.add_argument('--topk', type = int, default = 5,
                    help = 'Top number of classes to be predicted')

parser.add_argument('--cat_names', type = str, default = 'cat_to_name.json',
                    help = 'File containing category to name maps')
parser.add_argument('--model', type = str, default = 'vgg11', 
                    help = 'Model architecture from wither of VGG11 or Alexnet')

parser.add_argument('--hidden_units', type = int, default = 512,
                    help = 'Number of nodes in the hidden layer')

# function that loads a checkpoint 

args_in = parser.parse_args()

def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    
#     model = models.vgg11(pretrained=True)
    
#     for param in model.parameters():
#         param.requires_grad = False
        
#     model.classifier = nn.Sequential(
#           nn.Linear(25088, 512),
#           nn.ReLU(),
#           nn.Dropout(p=0.2),
#           nn.Linear(512, 256),
#           nn.ReLU(),
#           nn.Dropout(p=0.2),
#           nn.Linear(256, 102),
#           nn.LogSoftmax(dim = 1)
#         )

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

    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_checkpoint(args_in.checkpoint_directory)
model.to(device);

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image) 
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    img = np.array(transform(img))

    return img

# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
    
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.transpose((1, 2, 0))
    
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
    
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
    
#     ax.imshow(image)
    
#     return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.to('cuda')
    
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.float()
    
    with torch.no_grad():
        model.to(device)
        outputs = model.forward(image.cuda())
        probs, classes = torch.exp(outputs).topk(topk)
        return probs[0].tolist(), classes[0].add(1).tolist()
            
predict(args_in.img_path, model)

# def predict_graphs(img_path, model):
#     '''Predict the top 5 likelyhoods of a  picture using graphs
    
#     args:
#         img_path -> [str]: directory to an image
#         model -> [object]: a pretrained model.
        
#     returns:
#         images containing graphs and flowers
        
#     Part of this function was adopted from matplotlib.org
#     '''
#     plt.figure(figsize=(3,3))

#     #img_path = ('flowers/test/1/image_06743.jpg')

#     probs, cats = predict(img_path, model)
#     img = process_image(img_path)
#     index = cats[0]

#     ax = imshow(img, ax = plt)
#     ax.axis('off')
#     ax.title(args_in.cat_names[str(max_index)])
#     ax.show()

#     plt.figure(figsize=(3,3))

#     names = [args_in.cat_names[str(index)] for index in cats]
#     y_ax = np.arange(len(names))
#     proba = np.array(probs)

#     plt.barh(y_pos, proba, color='blue')
#     plt.yticks(y_ax, names)
#     plt.gca().invert_yaxis()  

probs, cats = predict(args_in.img_path, model, args_in.topk)


if args_in.cat_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[str(index)] for index in cats]
    print("Class name:")
    print(names)

print("Class number:")
print(cats)
print("Probability (%):")
for idx, item in enumerate(probs):
    probs[idx] = round(item*100, 2)
print(probs)

## Usage
'''NB: With the checkpoing being trained on alexnet, the predict.py script should be used only with the --model alexnet 
argunemt. Otherwise, a seperate checkpoint should be created for vgg.'''
# python predict.py flowers/test/1/image_06743.jpg --model alexnet