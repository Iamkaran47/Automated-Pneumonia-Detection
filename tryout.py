import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Define constants and paths
EPOCHS = 100  # Increase number of epochs for better training
data_dir = "chest_xray"
TEST = 'test'
TRAIN = 'train'


# Define function to count number of images in each folder
def count_images_in_folders(data_dir):
    folder_counts = {}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            folder_counts[folder] = len(os.listdir(folder_path))
    return folder_counts


# Define directory paths
train_data_dir = os.path.join(data_dir, TRAIN)
test_data_dir = os.path.join(data_dir, TEST)

# Count number of images in train and test folders
train_folder_counts = count_images_in_folders(train_data_dir)
test_folder_counts = count_images_in_folders(test_data_dir)

# Calculate total number of images in train and test sets
total_train_images = sum(train_folder_counts.values())
total_test_images = sum(test_folder_counts.values())


# Calculate batch size for train and test sets
def calculate_batch_size(total_images):
    # Aim for approximately 300 images per epoch
    target_images_per_epoch = 300
    # Calculate batch size
    batch_size = max(1, total_images // target_images_per_epoch)
    return batch_size


# Calculate batch size for train and test sets
train_batch_size = calculate_batch_size(total_train_images)
test_batch_size = calculate_batch_size(total_test_images)

print("Total Train Images:", total_train_images)
print("Total Test Images:", total_test_images)
print("Train Batch Size:", train_batch_size)
print("Test Batch Size:", test_batch_size)

# Define data transformations
data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),  # Additional data augmentation
        transforms.RandomAffine(degrees=0, shear=10),  # Additional data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

# Load datasets and create data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [TRAIN, TEST]}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=train_batch_size if x == TRAIN else test_batch_size,
                                   shuffle=True) for x in [TRAIN, TEST]}
dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}
class_names = image_datasets[TRAIN].classes

# Display sample images from the training set
fig, axes = plt.subplots(6, 6, figsize=(12, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i in range(6):
    for j in range(6):
        inputs, classes = next(iter(dataloaders[TRAIN]))
        input_img = inputs[0]
        class_label = classes[0]
        inp = input_img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        axes[i, j].imshow(inp)
        axes[i, j].set_title(class_names[class_label.item()])
        axes[i, j].axis('off')

plt.show()

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Send model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Learning rate scheduler
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in [TRAIN, TEST]:
            if phase == TRAIN:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == TRAIN:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == TEST and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best Test Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


# Train the fine-tuned model
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=EPOCHS)

import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_model(model, dataloader, average='macro'):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct_predictions / total_predictions

    # Calculate precision, recall, and F1-score using appropriate averaging for multiclass problems
    if average is not None:
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
    else:
        # Calculate per-class metrics
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')  # Handle potential multiclass case
    print(f'Recall: {recall:.4f}')  # Handle potential multiclass case
    print(f'F1-score: {f1:.4f}')  # Handle potential multiclass case


# Usage example (assuming dataloaders is a dictionary with 'TEST' key)
evaluate_model(model, dataloaders[TEST])
model.save('pneumonia_tryout.h5')
