import os
import torch
import time
import random
from torchvision import transforms, datasets
from PIL import Image
from model import DnCNN
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define a transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256 pixels
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    # Normalize to [0, 1] for RGB channels
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(
    './datasets/processed_data/eval_denoiser/td_test/clean', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = "./checkpoints/20231023_211535"
save_dir = "./datasets/processed_data/eval_denoiser/td_test/denoised"

model = DnCNN(channels=3)
model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
model.to(device)


def unnormalize(tensor, mean, std):
    """
    Reverses the normalization on a tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL image.
    """
    tensor = tensor.clone().detach()
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)
    tensor = (tensor * 255).astype('uint8')
    return Image.fromarray(tensor)


# Inference
model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, total=len(test_loader)):
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)

        # Save images
        for img, label in zip(inputs, labels):
            person_name = test_dataset.classes[label]
            # reverse normalization
            img = unnormalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            img = tensor_to_pil(img)

            person_dir = os.path.join(save_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            time_str = time.strftime("%Y%m%d_%H%M%S")
            img.save(os.path.join(
                person_dir, f'{time_str}_{random.randint(0, 100000)}.jpg'))
