# Count the total number of folders
total_folders=$(ls -d ./datasets/processed_data/* | wc -l)

# Calculate the number of folders to move to each destination
folders_to_move_train=$((total_folders / 2))
folders_to_move_eval=$((total_folders - folders_to_move_train))

# Create the destination folders
train_folder="./datasets/processed_data/train_denoiser"
eval_folder="./datasets/processed_data/eval_denoiser"
mkdir -p $train_folder
mkdir -p $eval_folder

# Initialize counters
counter_train=0
counter_eval=0

# Loop through each folder and move it
for dir in ./datasets/processed_data/*/; do
    # Move folders to 'train_denoiser' until reaching half the total number
    if [ $counter_train -lt $folders_to_move_train ]; then
        mv "$dir" "$train_folder/"
        ((counter_train++))
        # Move the remaining folders to 'eval_denoiser'
    elif [ $counter_eval -lt $folders_to_move_eval ]; then
        mv "$dir" "$eval_folder/"
        ((counter_eval++))
    fi
done
