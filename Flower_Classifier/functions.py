# This file contains all necessary functions and classes for running train/predict.py

import argparse

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import seaborn as sb


def load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(45),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])



    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

    trainloader = DataLoader(training_data, batch_size = 32, shuffle=True)
    validloader = DataLoader(validation_data, batch_size = 32)
    testloader = DataLoader(testing_data, batch_size = 32)

    return training_data, validation_data, testing_data, trainloader, validloader, testloader

def new_classifier(model):
    for param in model.parameters():
        param.requires_grad = False
        
        model.classifier = nn.Sequential(nn.Linear(25088, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(256, 102),
                                        nn.LogSoftmax(dim=1)
                                        )
        return model
