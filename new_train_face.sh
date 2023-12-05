set -e
set -x
export CUDA_VISIBLE_DEVICES=1
python -u new_train_face.py --mode denoised
