# Local working directory "/Users/t9349ch/Desktop/SWX/Udacity/Python Projects/Flower classifier/flowers"

import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from functions import load_data

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
gpu = args.GPU

training_data, validation_data, testing_data, trainloader, validloader, testloader = load_data(data_dir)

print(args.data_directory)
print(args.arch)
