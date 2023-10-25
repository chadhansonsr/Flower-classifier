# Local working directory "/Users/t9349ch/Desktop/SWX/Udacity/Python Projects/Flower classifier/flowers"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from functions import load_data, new_classifier, train_model, test_model, save_model

parser = argparse.ArgumentParser(description="Train a new neural network")

parser.add_argument("--data_directory", default="/Users/t9349ch/Desktop/SWX/Udacity/Python Projects/Flower classifier/flowers", action="store", help="path to data directory")
parser.add_argument("--save_directory", type=str, default="checkpoint.pth", help="path to save model")
parser.add_argument("--arch", type=str, default="vgg16", help="architecture (default: vgg16)")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate (default: 0.001)")
parser.add_argument("--hidden_units", type=int, default=256, help="hidden units (default: 256)")
parser.add_argument("--epochs", type=int, default=25, help="epochs (default: 15)")
parser.add_argument("--GPU", action="store_true", help="use GPU for training")

args = parser.parse_args()

data_directory = args.data_directory
save_directory = args.save_directory
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
GPU = args.GPU

training_data, validation_data, testing_data, trainloader, validloader, testloader = load_data(data_directory)

# Load pre-trained model
model = getattr(models, args.arch)(pretrained=True)

# Attach new classifier
new_classifier(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train model
model, optimizer = train_model(model, epochs, trainloader, validloader, optimizer, criterion, GPU)

# Test model
test_model(model, testloader, criterion, GPU)

# Save model
save_model(model, epochs, training_data)
