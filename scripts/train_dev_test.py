import os
import shutil
import random
import argparse

random.seed(0)
# Define the root directory and target directories
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str)
args = parser.parse_args()

train_dir = os.path.join(args.root_dir, 'train')
dev_dir = os.path.join(args.root_dir, 'dev')
test_dir = os.path.join(args.root_dir, 'test')

# Create the target directories if they do not exist
for target_dir in [train_dir, dev_dir, test_dir]:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# Iterate through each person's directory
for person_name in os.listdir(args.root_dir):
    person_path = os.path.join(args.root_dir, person_name)

    # Skip if not a directory or if it's one of the target directories
    if not os.path.isdir(person_path) or person_name in ['train', 'dev', 'test']:
        continue

    # List all jpg files
    jpg_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]

    # Shuffle and split the data
    random.shuffle(jpg_files)
    total_files = len(jpg_files)
    train_count = int(total_files * 0.8)
    dev_count = int(total_files * 0.1)

    train_files = jpg_files[:train_count]
    dev_files = jpg_files[train_count:train_count + dev_count]
    test_files = jpg_files[train_count + dev_count:]

    # Create new directories for this person in train, dev, and test folders
    for target_dir in [train_dir, dev_dir, test_dir]:
        new_person_dir = os.path.join(target_dir, person_name)
        if not os.path.exists(new_person_dir):
            os.makedirs(new_person_dir)

    # Move the files
    for f in train_files:
        shutil.move(os.path.join(person_path, f),
                    os.path.join(train_dir, person_name, f))
    for f in dev_files:
        shutil.move(os.path.join(person_path, f),
                    os.path.join(dev_dir, person_name, f))
    for f in test_files:
        shutil.move(os.path.join(person_path, f),
                    os.path.join(test_dir, person_name, f))
    # remove the empty directory
    os.rmdir(person_path)
