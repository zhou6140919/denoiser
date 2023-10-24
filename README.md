# Denoiser

## Preparation

1. Install [Fawkes](https://github.com/Shawn-Shan/fawkes)
    ```bash
    conda create -n denoiser python=3.7
    pip install -r requirements.txt
    ```
2. Download datasets to `datasets` folder, see `datasets/README.md`.
3. Rename `pubfig` folder name
    ```bash
    bash ./rename_pubfig.sh
    ```
4. Remove individuals with less than 20 images.
    ```bash
    python -u ./preprocess.py
    ```

    ```bash
                count
    count  638.000000
    mean    78.465517
    std     34.530007
    min      7.000000
    25%     59.000000
    50%     78.000000
    75%     96.000000
    max    403.000000
    num of people less than 20 images 0.0109717868338558 %
5. Split into `train_denoiser` and `eval_denoiser` 5:5.
    ```bash
    bash ./split.sh
    ```
6. Use `Fawkes` to cloak images.
    About 40 hrs.
    ```bash
    python ./fawkes.py
    ```
7. Deal with images.
    ```bash
    python ./prepare_for_train_denoiser.py
    ```
    Train: 17878
    Dev: 2216
    Test: 2581
8. Start training.
    ```bash
    bash ./train.sh
    ```


## Updates

- 09/08/2023 Project started.
- 09/26/2023 Download datasets.
- 10/20/2023 Preprocessing.
- 10/23/2023 Training.
