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
from functions import load_data, process_image, predict

parser = argparse.ArgumentParser(description="Predict flower name")

parser.add_argument("image_path", action="store", help="path to image")
parser.add_argument("--save_directory", type=str, default="checkpoint.pth", help="path to save model")
parser.add_argument("--arch", type=str, default="vgg16", help="architecture (default: vgg16)")
parser.add_argument("--top_k", type=int, default=5, help="most likely flower names (default: 5)")
parser.add_argument("--category_names", default="cat_to_name.json", help="map categories to names")
parser.add_argument("--GPU", action="store_true", help="use GPU for training")

args = parser.parse_args()

image = args.image_path
save_dir = args.save_directory
arch = args.arch
top_k = args.top_k
category_names = args.category_names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
GPU = args.GPU

# Load pre-trained model
model = getattr(models, args.arch)(pretrained=True)

# process image
processed_image = process_image(image)
if GPU == True:
    inputs, labels = processed_image.to("cuda")
else:
    pass

probabilities, classes = predict(processed_image, model, top_k, GPU)

print(probabilities, classes)