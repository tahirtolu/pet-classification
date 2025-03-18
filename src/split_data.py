import os
import shutil
import random

def split_data(source_dir, train_dir, valid_dir, test_dir):
    """
    Split the data into train, validation and test sets with fixed numbers:
    - Train: 1000 images per class
    - Validation: 200 images per class
    - Test: 200 images per class
    """
    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Fixed numbers for each split
    train_size = 1000
    valid_size = 200
    test_size = 200

    print(f"Target split sizes per class:")
    print(f"Train: {train_size}")
    print(f"Validation: {valid_size}")
    print(f"Test: {test_size}")

    # Get all animal classes
    animal_classes = [d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d)) and 
                     d not in ['train', 'valid', 'test']]

    for animal in animal_classes:
        print(f"\nProcessing {animal}...")
        
        # Create animal directories
        os.makedirs(os.path.join(train_dir, animal), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, animal), exist_ok=True)
        os.makedirs(os.path.join(test_dir, animal), exist_ok=True)

        # Get all images for this animal
        animal_dir = os.path.join(source_dir, animal)
        images = [f for f in os.listdir(animal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(images)
        print(f"Total images available: {total_images}")

        if total_images < (train_size + valid_size + test_size):
            print(f"Warning: Not enough images for {animal}. Skipping...")
            continue

        # Randomly shuffle images
        random.shuffle(images)

        # Split images
        train_images = images[:train_size]
        valid_images = images[train_size:train_size + valid_size]
        test_images = images[train_size + valid_size:train_size + valid_size + test_size]

        # Copy images to respective directories
        for img, dst_dir in [
            (train_images, train_dir),
            (valid_images, valid_dir),
            (test_images, test_dir)
        ]:
            for image in img:
                src = os.path.join(animal_dir, image)
                dst = os.path.join(dst_dir, animal, image)
                shutil.copy2(src, dst)

        print(f"Split completed for {animal}:")
        print(f"  Train: {len(train_images)}")
        print(f"  Valid: {len(valid_images)}")
        print(f"  Test: {len(test_images)}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Define directories
    source_dir = "Training Data"
    train_dir = os.path.join(source_dir, "train")
    valid_dir = os.path.join(source_dir, "valid")
    test_dir = os.path.join(source_dir, "test")

    # First, remove existing split directories if they exist
    for dir_path in [train_dir, valid_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed existing directory: {dir_path}")

    # Split data
    split_data(source_dir, train_dir, valid_dir, test_dir) 