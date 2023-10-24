from torch.utils.data import Dataset
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.clean_path = os.path.join(folder_path, 'clean')
        self.noisy_path = os.path.join(folder_path, 'noisy')
        self.transform = transform
        self.clean_images = [f for f in os.listdir(
            self.clean_path) if os.path.isfile(os.path.join(self.clean_path, f))]

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_img_path = os.path.join(self.clean_path, self.clean_images[idx])
        noisy_img_path = os.path.join(
            self.noisy_path, self.clean_images[idx].replace('.jpg', '_cloaked.png'))
        clean_img = Image.open(clean_img_path).convert('RGB')  # Convert to RGB
        noisy_img = Image.open(noisy_img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        return clean_img, noisy_img
