import os
import shutil

# Set the train and eval directories
train_dir = 'datasets/processed_data/train_denoiser'
eval_dir = 'datasets/processed_data/eval_denoiser'

# Create the train and eval directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
print(train_dir)

os.makedirs(eval_dir, exist_ok=True)
print(eval_dir)

# Initialize the count
count = 0

# Loop through each sub-directory in the processed_data directory
a = os.listdir('datasets/processed_data/')
a.sort()
for dir_name in a:
    dir_path = os.path.join('datasets/processed_data/', dir_name)
    if 'train_denoiser' in dir_path or 'eval_denoiser' in dir_path:
        continue

    # Check if it's a directory
    if os.path.isdir(dir_path):
        count += 1

        # Move the directory to train or eval based on the count
        if count < 316:
            shutil.move(dir_path, os.path.join(train_dir, dir_name))
        else:
            shutil.move(dir_path, os.path.join(eval_dir, dir_name))
