# import necessary libraries
import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from model import FaceRecognitionModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='mix',
                    choices=['mix', 'denoised', 'noisy'])
args = parser.parse_args()
print("Mode: ", end='')
print(args.mode)

# Data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('checkpoints', timestamp)
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(device)

if args.mode == 'mix':
    train_dir = './datasets/processed_data/eval_denoiser/new_train'
elif args.mode == 'denoised':
    train_dir = './datasets/processed_data/eval_denoiser/new_train_denoised'
elif args.mode == 'noisy':
    train_dir = './datasets/processed_data/eval_denoiser/new_train_noisy'
else:
    raise NotImplementedError(f"Mode {args.mode} not implemented")

train_dataset = datasets.ImageFolder(
    train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)

dev_dataset = datasets.ImageFolder(
    "./datasets/processed_data/eval_denoiser/new_dev", transform=transform)
dev_loader = DataLoader(dev_dataset, batch_size=48, shuffle=False)

test_datasets = datasets.ImageFolder(
    f'./datasets/processed_data/eval_denoiser/new_test', transform=transform)
test_loaders = DataLoader(
    test_datasets, batch_size=96, shuffle=False)

# Model
# model = models.resnet18(pretrained=True)
# model_name = 'resnet18'
model = models.alexnet(pretrained=True)
model_name = 'alexnet'

# model = models.resnet50(pretrained=True)
# model = models.densenet121(pretrained=True)
# model_name = 'densenet121'
print(f'Model: {model_name}')
# freeze_params = False
# if freeze_params:
#     logger.info("Freezing all layers except last")
#     for param in model.parameters():
#         param.requires_grad = False
#
# if model.__class__.__name__ == 'ResNet':
#     logger.info("Model: ResNet18")
#     if freeze_params:
#         for param in model.fc.parameters():
#             param.requires_grad = True
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, len(train_dataset.classes))
# elif model.__class__.__name__ == 'DenseNet':
#     logger.info("Model: densenet121")
#     if freeze_params:
#         for param in model.classifier.parameters():
#             param.requires_grad = True
#     num_features = model.classifier.in_features
#     model.classifier = nn.Linear(num_features, len(train_dataset.classes))
# elif model.__class__.__name__ == 'AlexNet':
#     logger.info("Model: AlexNet")
#     if freeze_params:
#         # For the first fully connected layer
#         for param in model.classifier[1].parameters():
#             param.requires_grad = True
#
#         # For the second fully connected layer
#         for param in model.classifier[4].parameters():
#             param.requires_grad = True
#
#         # For the third fully connected layer
#         for param in model.classifier[6].parameters():
#             param.requires_grad = True
#
#     num_features = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(num_features, len(train_dataset.classes))
# else:
#     logger.info("Model: Unknown")
logger.info(f"num_classes: {len(train_dataset.classes)}")

# model = FaceRecognitionModel(len(train_dataset.classes))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
model.to(device)

best_epoch = 0
best_acc = 0
num_epochs = 20
for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch}")
    logger.info("Training")
    model.train()

    pbar = tqdm(train_loader, total=len(train_loader), dynamic_ncols=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': loss.item()})
    logger.info("Evaluating")
    model.eval()
    pbar.close()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in tqdm(dev_loader, total=len(dev_loader), dynamic_ncols=True):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
        logger.info(f"Accuracy: {correct / total}")
        if correct / total > best_acc:
            best_acc = correct / total
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_dir, "best_model.pt"))

logger.info(f"Best Epoch: {best_epoch}")

# Testing
logger.info("Testing")
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loaders:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()
    logger.info(f"Accuracy: {correct / total}")
print(f'Model: {model_name}')
print(output_dir)
