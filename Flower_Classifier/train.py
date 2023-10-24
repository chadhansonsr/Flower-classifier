# Local working directory "/Users/t9349ch/Desktop/SWX/Udacity/Python Projects/Flower classifier/flowers"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from functions import load_data, new_classifier, train_model

parser = argparse.ArgumentParser(description="Train a new neural network")

parser.add_argument("data_directory", action="store", help="path to data directory")
parser.add_argument("--save_directory", type=str, default="checkpoint.pth", help="path to save model")
parser.add_argument("--arch", type=str, default="vgg16", help="architecture (default: vgg16)")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate (default: 0.001)")
parser.add_argument("--hidden_units", type=int, default=256, help="hidden units (default: 256)")
parser.add_argument("--epochs", type=int, default=15, help="epochs (default: 15)")
parser.add_argument("--GPU", action="store_true", help="use GPU for training")

args = parser.parse_args()

data_dir = args.data_directory
save_dir = args.save_directory
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
GPU = args.GPU

training_data, validation_data, testing_data, trainloader, validloader, testloader = load_data(data_dir)

# Load pre-trained model
model = getattr(models, args.arch)(weights="DEFAULT")

# Attach new classifier
new_classifier(model)

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train model
model, optimizer = train_model(model, epochs, trainloader, validloader, optimizer, criterion, GPU)

# print(args.data_directory)
# print(args.arch)
# print(args.epochs)
# print(model)
