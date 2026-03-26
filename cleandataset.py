import os
from PIL import Image
import hashlib
import shutil
from collections import Counter

# ======================
# STEP 1: FIND YOUR DATASET
# ======================
def find_dataset():
    possible_paths = [
        './chest_xray',
        './chest-xray-pneumonia',
        './data/chest_xray',
        './pneumonia',
        '../chest_xray',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Verify it has the right structure
            if os.path.exists(os.path.join(path, 'train')) or \
               os.path.exists(os.path.join(path, 'val')):
                print(f"✓ Found dataset at: {path}")
                return path
    
    print("Dataset not found. Please enter the full path:")
    return input("Path: ").strip()

# ======================
# STEP 2: ANALYZE STRUCTURE
# ======================
def analyze_structure(dataset_path):
    print("\n📊 Dataset Structure Analysis:")
    print("=" * 60)
    
    total_count = 0
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"\n📁 {split.upper()}/")
        
        for category in ['NORMAL', 'PNEUMONIA']:
            category_path = os.path.join(split_path, category)
            if os.path.exists(category_path):
                count = len([f for f in os.listdir(category_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  └── {category}: {count} images")
                total_count += count
    
    print(f"\n📊 TOTAL IMAGES: {total_count}")
    print("=" * 60)
    return total_count

# ======================
# STEP 3: CHECK CORRUPTED IMAGES
# ======================
def check_corrupted_images(dataset_path):
    print("\n🔍 Checking for corrupted images...")
    corrupted = []
    total = 0
    
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
                    total += 1
                    filepath = os.path.join(category_path, file)
                    try:
                        img = Image.open(filepath)
                        img.verify()
                        # Re-open for actual check
                        img = Image.open(filepath)
                        img.load()
                    except Exception as e:
                        corrupted.append(filepath)
                        print(f"  ✗ Corrupted: {filepath}")
                        print(f"    Error: {str(e)[:50]}")
    
    print(f"\n📊 Checked {total} images, found {len(corrupted)} corrupted")
    return corrupted

# ======================
# STEP 4: CHECK DUPLICATES
# ======================
def find_duplicates(dataset_path):
    print("\n🔍 Checking for duplicates...")
    hashes = {}
    duplicates = []
    
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
                    try:
                        with open(filepath, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        if file_hash in hashes:
                            duplicates.append(filepath)
                            print(f"  ✗ Duplicate of {hashes[file_hash]}")
                            print(f"    → {filepath}")
                        else:
                            hashes[file_hash] = filepath
                    except Exception as e:
                        print(f"  ⚠️ Error reading {filepath}: {e}")
    
    print(f"\n📊 Found {len(duplicates)} duplicate images")
    return duplicates

# ======================
# STEP 5: ANALYZE IMAGE PROPERTIES
# ======================
def analyze_images(dataset_path):
    print("\n🔍 Analyzing image properties...")
    sizes = []
    formats = []
    
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
                    try:
                        img = Image.open(filepath)
                        sizes.append(img.size)
                        formats.append(img.format)
                    except:
                        continue
    
    print(f"\n📊 Image Properties:")
    print(f"  Formats: {set(formats)}")
    print(f"  Unique sizes: {len(set(sizes))}")
    
    # Show most common sizes
    size_counts = Counter(sizes)
    print(f"\n  Most common sizes:")
    for size, count in size_counts.most_common(5):
        print(f"    {size}: {count} images")
    
    return sizes, formats

# ======================
# STEP 6: CLEAN DATASET
# ======================
def clean_dataset(dataset_path, corrupted, duplicates):
    # Create bad_images folder with same structure
    bad_images_path = os.path.join(os.path.dirname(dataset_path), 'bad_images')
    
    print(f"\n🧹 Creating folder: {bad_images_path}")
    
    all_bad = corrupted + duplicates
    moved = 0
    
    for filepath in all_bad:
        if os.path.exists(filepath):
            # Recreate folder structure
            rel_path = os.path.relpath(filepath, dataset_path)
            dest_path = os.path.join(bad_images_path, rel_path)
            
            # Create destination folder
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Move file
            shutil.move(filepath, dest_path)
            moved += 1
            print(f"  ✓ Moved: {os.path.basename(filepath)}")
    
    print(f"\n✅ Moved {moved} bad images to {bad_images_path}")

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    print("=" * 60)
    print("      PNEUMONIA DATASET CLEANER")
    print("=" * 60)
    
    # Step 1: Find dataset
    dataset_path = find_dataset()
    
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset not found. Exiting.")
        exit()
    
    # Step 2: Show structure
    total = analyze_structure(dataset_path)
    
    # Step 3: Analyze images
    analyze_images(dataset_path)
    
    # Step 4: Find problems
    corrupted = check_corrupted_images(dataset_path)
    duplicates = find_duplicates(dataset_path)
    
    # Step 5: Summary and cleanup
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  ✓ Total images: {total}")
    print(f"  ✗ Corrupted: {len(corrupted)}")
    print(f"  ✗ Duplicates: {len(duplicates)}")
    print(f"  ✅ Clean: {total - len(corrupted) - len(duplicates)}")
    print("=" * 60)
    
    # Step 6: Ask to clean
    if corrupted or duplicates:
        print(f"\n⚠️ Found {len(corrupted) + len(duplicates)} problematic images")
        choice = input("Move them to 'bad_images' folder? (yes/no): ")
        
        if choice.lower() in ['yes', 'y']:
            clean_dataset(dataset_path, corrupted, duplicates)
            print("\n✅ Dataset cleaned!")
            analyze_structure(dataset_path)  # Show new counts
        else:
            print("\n⏭️ Skipped cleaning")
    else:
        print("\n🎉 Dataset is already clean! No issues found.")
    
    print("\n" + "=" * 60)
    print("DONE! Your dataset is ready for training.")
    print("=" * 60)
