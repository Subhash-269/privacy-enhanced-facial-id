import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = ImageFolder(root='face_recog-dataset/', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model and modify final layer for 28 classes
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 28)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/privacy-facial-recog_2')

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
        # Log training loss every 10 mini-batches
        if i % 10 == 9:
            avg_loss = running_train_loss / 10
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/Train', avg_loss, global_step)
            running_train_loss = 0.0

    # Validation loop (loss logging)
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')

print('Finished Training')
torch.save(model.state_dict(), 'FRT-No_encryption.pth')
print('Model saved to FRT-No_encryption.pth')

# ------------------ Evaluation Metrics ------------------ #
# Collect predictions and true labels (and probabilities for ROC)
model.eval()
all_labels = []
all_preds = []
all_probs = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Compute accuracy, precision, and F1 score (weighted for multiclass)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

writer.add_scalar("Eval/Accuracy", accuracy)
writer.add_scalar("Eval/Precision", precision)
writer.add_scalar("Eval/F1", f1)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision (weighted): {precision:.4f}')
print(f'F1 Score (weighted): {f1:.4f}')

# Compute and save Confusion Matrix
# Compute and save Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
class_names = dataset.classes  # Get the class names from the dataset

fig_cm, ax_cm = plt.subplots(figsize=(12, 12))
im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax_cm.figure.colorbar(im, ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
ax_cm.set_ylabel("True label")
ax_cm.set_xlabel("Predicted label")

# Set tick labels using the class names
ax_cm.set_xticks(np.arange(len(class_names)))
ax_cm.set_xticklabels(class_names, rotation=45, ha='right')
ax_cm.set_yticks(np.arange(len(class_names)))
ax_cm.set_yticklabels(class_names)

plt.tight_layout()
writer.add_figure("Confusion Matrix", fig_cm)
fig_cm.savefig("confusion_matrix.jpg", format="jpg")
plt.close(fig_cm)


# Compute ROC curves (one-vs-rest) for multiclass and save the figure
num_classes = 28
# Binarize the true labels for ROC computation
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
for i in range(num_classes):
    ax_roc.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve (One-vs-Rest)')
ax_roc.legend(loc="lower right", fontsize='small')
plt.tight_layout()
writer.add_figure("ROC Curve", fig_roc)
fig_roc.savefig("roc_curve.jpg", format="jpg")
plt.close(fig_roc)

# ------------------ Bias Evaluation Placeholder ------------------ #
# Note: Bias evaluation metrics require subgroup (demographic) labels.
# If such labels are available, you can compute fairness metrics (e.g., disparate impact, equal opportunity)
# for each subgroup and log them similarly to the metrics above.
#
# For example:
# subgroup_metrics = compute_bias_metrics(all_labels, all_preds, subgroup_labels)
# writer.add_scalar("Bias/Metric_Name", subgroup_metrics["Metric_Name"], global_step)

# Close the TensorBoard writer
writer.close()
