for dir in ./datasets/pubfig/*; do
    # Only proceed if it's a directory
    if [ -d "$dir" ]; then
        # Replace spaces with underscores
        new_dir=$(echo "$dir" | tr ' ' '_')
        # Rename the directory
        mv "$dir" "$new_dir"
    fi
done
