"""
This script is used to prepare the data in `eval_denoiser` for training the facial recognition model and evaluating denoiser.
Mixing the clean and noisy images in the same folder without duplicates.
"""

import os
from glob import glob
import shutil
from tqdm import tqdm
import random

root_dir = './datasets/processed_data/eval_denoiser/'


for mode in ['train', 'dev', 'test']:
    print("Processing", mode)
    new_dir = os.path.join(root_dir, f'new_{mode}')
    os.makedirs(new_dir, exist_ok=True)
    curr_dir = os.path.join(root_dir, mode)
    for person in tqdm(glob(curr_dir + '/*/'), total=len(glob(curr_dir + '/*/'))):
        new_person_dir = os.path.join(new_dir, person.split('/')[-2])
        os.makedirs(new_person_dir, exist_ok=True)
        person_img_count = 0
        for file in glob(person + '*.jpg'):
            # print(file)
            if file.replace('.jpg', '_cloaked.png') in glob(person + '*.png'):
                if random.random() < 0.5:
                    shutil.copy(file, new_person_dir)
                else:
                    shutil.copy(file.replace(
                        '.jpg', '_cloaked.png'), new_person_dir)
                person_img_count += 1
        if person_img_count == 0:
            print(f"No cloaked images found for {person}")
