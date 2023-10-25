# This file contains all necessary functions and classes for running train/predict.py

import argparse

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

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
import os



def load_data():
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
    print_every = 50
    torch.cuda.empty_cache()

    if GPU and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            
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


def test_model(model, testloader, criterion, GPU):
    test_loss = 0
    accuracy = 0
    torch.cuda.empty_cache()

    if GPU and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    with torch.no_grad():
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)
            
            test_loss += batch_loss.item()
            probs = torch.exp(outputs)
            top_p, top_class = probs.topk(1, dim=1)
            compare = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(compare.type(torch.FloatTensor)).item()

    model.train()

    print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

def save_model(model, epochs, training_data):
    checkpoint = {"state_dict": model.state_dict(),
                "classifier": model.classifier,
                "epochs": epochs,
                "class_to_idx": training_data.class_to_idx
                }

    checkpoint_path = os.path.join(".", "checkpoint.pth")
    torch.save(checkpoint, "checkpoint.pth")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    pil = transform(image)
    array = np.array(pil)

    return array

def predict(image_path, model, top_k, GPU):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if GPU and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()

    with torch.no_grad():
        output = model.forward(image_tensor.cuda())
    probs = torch.exp(output)

    # Get the topk probabilities and indices
    top_probs, top_indices = probs.topk(top_k)
    
    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    return top_probs[0].tolist(), top_classes
