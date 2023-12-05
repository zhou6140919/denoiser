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

## Train a Facial Recognition Model

1. Data preparation
    ```bash
    python ./prepare_for_train_face.py
    ```
2. Prepare denoised images for evaluation.
    ```bash
    python ./inference.py
    ```
3. Training
    ```bash
    bash ./train_face.sh
    ```

## Practical Scenarios

We would like to imitate the real image distributions on the Internet.
So we mix noisy and clean image with a ratio of 0.5 randomly for our baseline.

```bash
python ./prepare_new_eval_dataset.py
```

Then we have to get the denoised images.
```bash
python ./new_inference.py
```

Next, we get all noisy images.
```bash
python ./new_noisy.py
```

Train facial recognition model on different training sets.
Change the `--mode` parameter to select datasets.

```bash
bash ./new_train_face.sh
```
You will get testing results on mixed dataset.

Testing on clean dataset.
Change checkpoint name in `test_face.py` before testing.

```bash
bash ./test_face.sh
```


## Updates

- 09/08/2023 Project started.
- 09/26/2023 Download datasets.
- 10/20/2023 Preprocessing.
- 10/23/2023 Training.
- 11/20/2023 More Training.
- 11/28/2023 Presentation.
- 12/05/2023 Report+Code.
