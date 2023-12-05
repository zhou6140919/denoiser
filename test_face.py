import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Testing")
model = models.alexnet(pretrained=True)
output_dir = os.path.join('checkpoints', '20231127_133836')
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
model.eval()
test_datasets = datasets.ImageFolder(
    f'./datasets/processed_data/eval_denoiser/td_test/clean', transform=transform)
test_loaders = DataLoader(
    test_datasets, batch_size=96, shuffle=False)
model.to(device)
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in tqdm(test_loaders, total=len(test_loaders)):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()
    print(f"Accuracy: {correct / total}")
print(output_dir)
