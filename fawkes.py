import os
from tqdm import tqdm

for d in ['train', 'eval']:
    print("Running on large {} set".format(d))
    for mode in ['train', 'dev', 'test']:
        print("Running on {} set".format(mode))
        people_names = os.listdir(
            f'./datasets/processed_data/{d}_denoiser/{mode}')
        for name in tqdm(people_names, total=len(people_names)):
            os.system("""
                fawkes -d "{}" --mode low -g 0 --batch-size 32
            """.format(os.path.join(f'./datasets/processed_data/{d}_denoiser/{mode}', name)))
