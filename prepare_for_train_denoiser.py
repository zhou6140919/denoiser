"""
This script is used to prepare the data in `train_denoiser` for training the denoiser.
"""
import os
from glob import glob
import shutil
from tqdm import tqdm

root_dir = './datasets/processed_data/train_denoiser/'


for mode in ['train', 'dev', 'test']:
    print("Processing", mode)
    new_dir = os.path.join(root_dir, f'td_{mode}')
    new_dir_clean = os.path.join(new_dir, 'clean')
    new_dir_noisy = os.path.join(new_dir, 'noisy')
    os.makedirs(new_dir_clean, exist_ok=True)
    os.makedirs(new_dir_noisy, exist_ok=True)
    curr_dir = os.path.join(root_dir, mode)
    for person in tqdm(glob(curr_dir + '/*/'), total=len(glob(curr_dir + '/*/'))):
        # print(person)
        person_img_count = 0
        for file in glob(person + '*.jpg'):
            # print(file)
            if file.replace('.jpg', '_cloaked.png') in glob(person + '*.png'):
                shutil.copy(file, new_dir_clean)
                shutil.copy(file.replace(
                    '.jpg', '_cloaked.png'), new_dir_noisy)
                person_img_count += 1
        if person_img_count == 0:
            print(f"No cloaked images found for {person}")
