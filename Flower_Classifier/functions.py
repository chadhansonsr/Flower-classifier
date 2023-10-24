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

# def device():
#     return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def train_model(model, epochs, trainloader, validloader, optimizer, criterion, GPU):
    steps = 0
    running_loss = 0
    print_every = 25
    torch.cuda.empty_cache()

    if GPU == True:
        model.to("cuda")
    else:
        pass

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            
            if GPU == True:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            else:
                pass
            
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to("cuda"), labels.to("cuda")
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)

                        test_loss += batch_loss.item()

                        probs = torch.exp(outputs)
                        top_p, top_class = probs.topk(1, dim=1)
                        compare = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(compare.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.."
                    f"Training loss: {running_loss/print_every:.3f}.."
                    f"Test loss: {test_loss/len(validloader):.3f}.."
                    f"Test accuracy: {accuracy/len(validloader):.3f}.."
                    )
                running_loss = 0
                model.train()
    return model, optimizer

