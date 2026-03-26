import os
import shutil
import random
from collections import defaultdict

# ======================
# CONFIGURATION
# ======================
DATASET_PATH = './chest_xray'
OUTPUT_PATH = './chest_xray_balanced'

# Split ratios
TRAIN_RATIO = 0.70  # 70%
VAL_RATIO = 0.15    # 15%
TEST_RATIO = 0.15   # 15%

# ======================
# STEP 1: COLLECT ALL IMAGES
# ======================
def collect_all_images(dataset_path):
    """Collect all images from train/val/test folders"""
    print("\n📂 Collecting all images...")
    
    images_by_class = {
        'NORMAL': [],
        'PNEUMONIA': []
    }
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        for category in ['NORMAL', 'PNEUMONIA']:
            category_path = os.path.join(split_path, category)
            if not os.path.exists(category_path):
                continue
                
            for file in os.listdir(category_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(category_path, file)
                    images_by_class[category].append(filepath)
    
    print(f"\n✓ Found images:")
    print(f"  NORMAL: {len(images_by_class['NORMAL'])} images")
    print(f"  PNEUMONIA: {len(images_by_class['PNEUMONIA'])} images")
    print(f"  TOTAL: {len(images_by_class['NORMAL']) + len(images_by_class['PNEUMONIA'])} images")
    
    return images_by_class

# ======================
# STEP 2: REMOVE DUPLICATES
# ======================
def remove_duplicates_from_list(image_list):
    """Remove duplicate images based on filename"""
    print("\n🔍 Checking for duplicates in collected images...")
    
    import hashlib
    seen_hashes = {}
    unique_images = []
    duplicates = 0
    
    for filepath in image_list:
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash not in seen_hashes:
                seen_hashes[file_hash] = filepath
                unique_images.append(filepath)
            else:
                duplicates += 1
        except:
            continue
    
    if duplicates > 0:
        print(f"  ✓ Removed {duplicates} duplicates")
    
    return unique_images

# ======================
# STEP 3: CREATE BALANCED SPLIT
# ======================
def create_balanced_split(images_by_class, train_ratio, val_ratio, test_ratio):
    """Split images into train/val/test with balanced classes"""
    print("\n✂️ Creating balanced split...")
    
    splits = {
        'train': {'NORMAL': [], 'PNEUMONIA': []},
        'val': {'NORMAL': [], 'PNEUMONIA': []},
        'test': {'NORMAL': [], 'PNEUMONIA': []}
    }
    
    for category in ['NORMAL', 'PNEUMONIA']:
        # Remove duplicates
        images = remove_duplicates_from_list(images_by_class[category])
        
        # Shuffle for random split
        random.seed(42)  # For reproducibility
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits['train'][category] = images[:train_end]
        splits['val'][category] = images[train_end:val_end]
        splits['test'][category] = images[val_end:]
    
    # Print split summary
    print("\n📊 Split Summary:")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        normal_count = len(splits[split]['NORMAL'])
        pneumonia_count = len(splits[split]['PNEUMONIA'])
        total = normal_count + pneumonia_count
        balance = min(normal_count, pneumonia_count) / max(normal_count, pneumonia_count) * 100
        
        print(f"\n{split.upper()}:")
        print(f"  NORMAL: {normal_count}")
        print(f"  PNEUMONIA: {pneumonia_count}")
        print(f"  Total: {total}")
        print(f"  Balance: {balance:.1f}% (100% = perfectly balanced)")
    
    print("=" * 60)
    
    return splits

# ======================
# STEP 4: COPY FILES TO NEW STRUCTURE
# ======================
def copy_files_to_new_structure(splits, output_path):
    """Copy files to new directory structure"""
    print(f"\n📁 Creating new dataset at: {output_path}")
    
    # Remove old output if exists
    if os.path.exists(output_path):
        print(f"  ⚠️ Removing existing folder: {output_path}")
        shutil.rmtree(output_path)
    
    # Create new structure
    for split in ['train', 'val', 'test']:
        for category in ['NORMAL', 'PNEUMONIA']:
            dir_path = os.path.join(output_path, split, category)
            os.makedirs(dir_path, exist_ok=True)
    
    # Copy files
    print("\n📋 Copying files...")
    total_copied = 0
    
    for split in ['train', 'val', 'test']:
        for category in ['NORMAL', 'PNEUMONIA']:
            print(f"  Copying {split}/{category}...", end=" ")
            
            dest_dir = os.path.join(output_path, split, category)
            files = splits[split][category]
            
            for i, src_file in enumerate(files):
                filename = os.path.basename(src_file)
                dest_file = os.path.join(dest_dir, filename)
                
                # Handle duplicate filenames
                counter = 1
                while os.path.exists(dest_file):
                    name, ext = os.path.splitext(filename)
                    dest_file = os.path.join(dest_dir, f"{name}_copy{counter}{ext}")
                    counter += 1
                
                shutil.copy2(src_file, dest_file)
                total_copied += 1
            
            print(f"✓ ({len(files)} files)")
    
    print(f"\n✅ Successfully copied {total_copied} files!")

# ======================
# STEP 5: VERIFY NEW DATASET
# ======================
def verify_dataset(output_path):
    """Verify the new dataset structure"""
    print("\n🔍 Verifying new dataset...")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_path, split)
        print(f"\n📁 {split.upper()}/")
        
        for category in ['NORMAL', 'PNEUMONIA']:
            category_path = os.path.join(split_path, category)
            if os.path.exists(category_path):
                count = len([f for f in os.listdir(category_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  └── {category}: {count} images")
    
    print("=" * 60)

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("=" * 60)
    print("    DATASET RE-SPLITTER & BALANCER")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Dataset not found at: {DATASET_PATH}")
        print("Please check the path and try again.")
        exit()
    
    print(f"\n📂 Input dataset: {DATASET_PATH}")
    print(f"📂 Output dataset: {OUTPUT_PATH}")
    print(f"\n📊 Target split:")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Val:   {VAL_RATIO*100:.0f}%")
    print(f"  Test:  {TEST_RATIO*100:.0f}%")
    
    # Confirm before proceeding
    print("\n" + "=" * 60)
    choice = input("Continue? This will create a NEW balanced dataset (yes/no): ")
    
    if choice.lower() not in ['yes', 'y']:
        print("\n⏭️ Cancelled")
        exit()
    
    # Step 1: Collect all images
    images_by_class = collect_all_images(DATASET_PATH)
    
    # Step 2: Create balanced split
    splits = create_balanced_split(images_by_class, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Step 3: Copy files to new structure
    copy_files_to_new_structure(splits, OUTPUT_PATH)
    
    # Step 4: Verify
    verify_dataset(OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"\nYour balanced dataset is ready at: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("1. Verify the new dataset looks correct")
    print("2. Update your training script to use: './chest_xray_balanced'")
    print("3. You can delete the old dataset once you're happy with the new one")
    print("\n" + "=" * 60)