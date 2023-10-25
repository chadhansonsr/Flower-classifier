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
from functions import load_data, process_image, predict, new_classifier

parser = argparse.ArgumentParser(description="Predict flower name")

parser.add_argument("--image_path", default="/Users/t9349ch/Desktop/SWX/Udacity/Python Projects/Flower classifier/Flower_Classifier/flowers/test/2/image_05109.jpg", action="store", help="path to image")
parser.add_argument("--save_directory", type=str, default="checkpoint.pth", help="path to save model")
parser.add_argument("--arch", type=str, default="vgg16", help="architecture (default: vgg16)")
parser.add_argument("--top_k", type=int, default=5, help="most likely flower names (default: 5)")
parser.add_argument("--category_names", default="cat_to_name.json", help="map categories to names")
parser.add_argument("--GPU", action="store_true", help="use GPU for training")

args = parser.parse_args()

image_path = args.image_path
save_directory = args.save_directory
arch = args.arch
top_k = args.top_k
category_names = args.category_names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
use_GPU = args.GPU

# Load pre-trained model
model = getattr(models, args.arch)(pretrained=True)
model = new_classifier(model)

# Load image using PIL
image = Image.open(image_path)

# Process image
processed_image = process_image(image)
processed_image = torch.from_numpy(processed_image).unsqueeze(0)

if use_GPU:
    model.to("cuda")
    processed_image = processed_image.to("cuda")
else:
    model.to("cpu")
    processed_image = processed_image.to("cpu")

# Convert the processed image tensor to a numpy array
processed_image = transforms.ToPILImage()(processed_image.squeeze(0))

# Load checkpoint
checkpoint = torch.load(args.save_directory)
model.load_state_dict(checkpoint["state_dict"])
model.class_to_idx = checkpoint["class_to_idx"]

# Make prediction
probabilities, classes = predict(processed_image, model, top_k, use_GPU)

# Class index to name
class_names = [cat_to_name[str(class_index)] for class_index in classes]

print("Probabilities:", probabilities)
print("Classes:", class_names)
