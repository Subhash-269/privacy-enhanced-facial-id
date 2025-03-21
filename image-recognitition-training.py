import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = ImageFolder(root='Five_Faces/', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the final layer to match the number of classes (5 in this case)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

# Move the model to the GPU
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/FRT')

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()  # Set model to training mode
    running_train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
        # Log the training loss
        if i % 10 == 9:  # Log every 10 mini-batches
            writer.add_scalar('Loss/Train', running_train_loss / 10, epoch * len(train_loader) + i)
            running_train_loss = 0.0

    # Validation loop
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    print(f'Epoch {epoch+1}, Training Loss: {running_train_loss/len(train_loader)}, Validation Loss: {avg_val_loss}')

print('Finished Training')
torch.save(model.state_dict(), 'FRT-No_encryption.pth')
print('Model saved to FRT-No_encryption.pth')

# Close the TensorBoard writer
writer.close()