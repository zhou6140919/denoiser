import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import sys
from PIL import ImageOps
from model import DenoiseCNN, DnCNN
from torch import optim, nn
from tqdm import tqdm
import torch
from torchvision import transforms
from dataset import CustomImageDataset
import logging
import time


timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('checkpoints', timestamp)
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(device)

# Define a transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 pixels
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # Normalize to [0, 1] for RGB channels
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load datasets
logger.info("Loading datasets...")
# Create datasets using the custom class
train_dataset = CustomImageDataset(
    folder_path='./datasets/processed_data/train_denoiser/td_train', transform=transform)
dev_dataset = CustomImageDataset(
    folder_path='./datasets/processed_data/train_denoiser/td_dev', transform=transform)
test_dataset = CustomImageDataset(
    folder_path='./datasets/processed_data/train_denoiser/td_test', transform=transform)


# Create data loaders
logger.info("Creating data loaders...")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=48, shuffle=True)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=48, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=48, shuffle=False)

# Initialize model, loss, and optimizer
logger.info("Initializing model, loss, and optimizer...")
model = DnCNN(channels=3)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)
# Number of epochs
n_epochs = 10

# Lists to keep track of training progress
train_loss_history = []
val_loss_history = []


best_dev_loss = np.inf

for epoch in range(n_epochs):
    # Training
    logger.info(f"Epoch {epoch}")
    logger.info("Training")
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    for clean_imgs, noisy_imgs in pbar:
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy_imgs.size(0)
        pbar.set_postfix(loss=loss.item())

    train_loss = train_loss / len(train_loader.dataset)
    train_loss_history.append(train_loss)

    # Validation
    logger.info("Validation")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for clean_imgs, noisy_imgs in tqdm(dev_loader, total=len(dev_loader)):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            val_loss += loss.item() * noisy_imgs.size(0)

    val_loss = val_loss / len(dev_loader.dataset)
    val_loss_history.append(val_loss)
    if val_loss < best_dev_loss:
        logger.info(f"Saving model at epoch {epoch}")
        best_dev_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(output_dir, 'best_model.pt'))

    logger.info(
        f"Epoch {epoch} - Train loss: {train_loss}, Validation loss: {val_loss}")


# Initialize test loss
test_loss = 0.0

# Set the model to evaluation mode
logger.info("Testing")
model.eval()

# Loop through the test set
with torch.no_grad():
    for clean_imgs, noisy_imgs in tqdm(test_loader, total=len(test_loader)):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # Forward pass
        outputs = model(noisy_imgs)

        # Compute loss
        loss = criterion(outputs, clean_imgs)

        # Update test loss
        test_loss += loss.item() * noisy_imgs.size(0)

# Compute the average test loss
test_loss = test_loss / len(test_loader.dataset)
logger.info(f"Test loss: {test_loss}")
logger.info(f"Best epoch: {np.argmin(val_loss_history)}")

# TODO: Change Location
# Pick some random test images
num_imgs = 4

test_subset = torch.utils.data.Subset(test_dataset, indices=np.random.choice(
    len(test_dataset), num_imgs, replace=False))

random_test_loader = torch.utils.data.DataLoader(
    test_subset, batch_size=num_imgs, shuffle=False)

# Get a batch from the random loader
noisy_imgs, clean_imgs = next(iter(random_test_loader))
noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

model = DnCNN(channels=3)
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
model.to(device)

# Denoise the images
with torch.no_grad():
    denoised_imgs = model(noisy_imgs)

noisy_imgs = noisy_imgs.cpu().numpy()
denoised_imgs = denoised_imgs.cpu().numpy()
clean_imgs = clean_imgs.cpu().numpy()
# Plot the images
fig, axes = plt.subplots(num_imgs, 3, figsize=(12, 12))

for i in range(num_imgs):
    axes[i, 0].imshow(np.transpose(noisy_imgs[i].squeeze(), (1, 2, 0)))
    axes[i, 0].set_title("Noisy")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(np.transpose(denoised_imgs[i].squeeze(), (1, 2, 0)))
    axes[i, 1].set_title("Denoised")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(np.transpose(clean_imgs[i].squeeze(), (1, 2, 0)))
    axes[i, 2].set_title("Clean")
    axes[i, 2].axis('off')

fig.savefig(os.path.join(output_dir, 'test_results.png'))
logger.info("Done!")
