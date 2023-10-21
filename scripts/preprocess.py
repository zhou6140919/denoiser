import os
import json
from glob import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool


# count cpu cores
cpu_count = os.cpu_count()
print(f'cpu count: {cpu_count}')
CPU_COUNT = 32 if cpu_count > 32 else cpu_count

# statistics of the dataset

# Initialize an empty dictionary to store the counts
image_counts = {}

# List of root directories
root_dirs = ['datasets/facescrub/actor_faces',
             'datasets/facescrub/actress_faces',
             'datasets/pubfig']

all_images = {}

# Iterate through the root directories
for root_dir in root_dirs:
    # Check if the directory exists
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        continue

    # Get the list of all person-wise directories in the root directory
    person_dirs = [d for d in os.listdir(
        root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Iterate through the person-wise directories
    for person in person_dirs:
        # Generate the glob pattern for .jpeg or .jpg files for the person
        pattern = os.path.join(root_dir, person, '*.jp*g')

        # Count the number of .jpeg files using glob
        num_images = len(glob(pattern))

        # Store the count in the dictionary
        image_counts[person] = num_images
        images = glob(pattern)
        images = [i for i in images if '(' not in i]
        if person in all_images:
            continue
        if len(images) > 20:
            all_images[person] = images

# with open('./a.json', 'w') as w:
#    json.dump(all_images, w, indent=4)

# Display the counts
# get min and max counts
df = pd.DataFrame.from_dict(image_counts, orient='index')
df.columns = ['count']
print(df.describe())
# find how many percent is lower than 20
print('num of people less than 20 images',
      len(df[df['count'] < 20]) / len(df), '%')
print('removing people less than 20 images')


def copy_image(person):
    new_dir = os.path.join('datasets', 'processed_data', person)
    os.makedirs(new_dir, exist_ok=True)
    images = all_images[person]
    for image in images:
        new_image = os.path.join(
            new_dir, os.path.basename(image).replace('.jpeg', '.jpg'))
        os.system('cp \"{}\" \"{}\"'.format(image, new_image))

    return person


# save new datasets, copy all images to new folder and to jpg format
# use multiprocessing to speed up
pool = Pool(CPU_COUNT)
with pool:
    s = list(
        tqdm(pool.imap(copy_image, all_images.keys()), total=len(all_images.keys())))
