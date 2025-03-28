import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 28)

# Load the saved model weights
model.load_state_dict(torch.load('FRT-No_encryption.pth'))
model = model.to(device)

# Load dataset to get class names
dataset = ImageFolder(root='face_recog-dataset/', transform=transform)

# Inference example
def predict_image(image_path):
    model.eval()  # Set the model to evaluation mode
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Transform and move to GPU
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return dataset.classes[predicted.item()]

# Example usage
image_path = 'test_images\images.jpeg'
# image_path = "test_images\modi-black-mr1.jpg"
# image_path = "test_images\images (1).jpeg"
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')