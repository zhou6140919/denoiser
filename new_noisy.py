import os
from glob import glob
import shutil
from tqdm import tqdm

root_dir = './datasets/processed_data/eval_denoiser/'


new_dir = os.path.join(root_dir, f'new_train_noisy')
os.makedirs(new_dir, exist_ok=True)
curr_dir = os.path.join(root_dir, 'train')
for person in tqdm(glob(curr_dir + '/*/'), total=len(glob(curr_dir + '/*/'))):
    new_person_dir = os.path.join(new_dir, person.split('/')[-2])
    os.makedirs(new_person_dir, exist_ok=True)
    person_img_count = 0
    for file in glob(person + '*.jpg'):
        # print(file)
        if file.replace('.jpg', '_cloaked.png') in glob(person + '*.png'):
            shutil.copy(file.replace(
                '.jpg', '_cloaked.png'), new_person_dir)
            person_img_count += 1
    if person_img_count == 0:
        print(f"No cloaked images found for {person}")
