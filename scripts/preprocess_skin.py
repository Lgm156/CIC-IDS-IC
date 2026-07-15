import pandas as pd
import numpy as np
import os
import glob
import joblib

def preprocess_skin():
    print("Preprocessing Skin Cancer Dataset...")
    
    metadata_path = "SkinCancerDataset/HAM10000_metadata.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata. Shape: {df.shape}")
    
    # Locate all images and map image_id to path
    part1_dir = "SkinCancerDataset/HAM10000_images_part_1"
    part2_dir = "SkinCancerDataset/HAM10000_images_part_2"
    
    image_paths = {}
    if os.path.exists(part1_dir):
        for filename in os.listdir(part1_dir):
            if filename.endswith(".jpg"):
                img_id = os.path.splitext(filename)[0]
                image_paths[img_id] = os.path.join(part1_dir, filename)
                
    if os.path.exists(part2_dir):
        for filename in os.listdir(part2_dir):
            if filename.endswith(".jpg"):
                img_id = os.path.splitext(filename)[0]
                image_paths[img_id] = os.path.join(part2_dir, filename)
            
    print(f"Mapped {len(image_paths)} images from part 1 and part 2 folders.")
    
    # Add paths to dataframe
    df['image_path'] = df['image_id'].map(image_paths)
    
    # Drop rows without image paths (if any)
    missing_images = df['image_path'].isna().sum()
    if missing_images > 0:
        print(f"Warning: {missing_images} metadata records have no matching image files.")
        df = df.dropna(subset=['image_path'])
        
    # Map dx to integers
    classes = sorted(df['dx'].unique())
    print(f"Found {len(classes)} classes: {classes}")
    
    dx_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    df['dx_idx'] = df['dx'].map(dx_to_idx)
    
    # Perform Lesion-Level Group splitting
    lesion_df = df.groupby('lesion_id').agg({
        'dx': 'first',
        'dx_idx': 'first',
        'image_id': 'count'
    }).reset_index()
    
    print("\nLesions per class:")
    print(lesion_df['dx'].value_counts())
    
    # Reserve validation and test lesion IDs
    # Target 30 images per class for validation and 30 for test.
    np.random.seed(42)
    
    val_lesions = set()
    test_lesions = set()
    
    for cls in classes:
        cls_lesions = lesion_df[lesion_df['dx'] == cls]['lesion_id'].values.copy()
        np.random.shuffle(cls_lesions)
        
        # Accumulate validation images
        val_img_count = 0
        val_idx = 0
        while val_img_count < 30 and val_idx < len(cls_lesions):
            lid = cls_lesions[val_idx]
            val_lesions.add(lid)
            val_img_count += len(df[df['lesion_id'] == lid])
            val_idx += 1
            
        # Accumulate test images
        test_img_count = 0
        while test_img_count < 30 and val_idx < len(cls_lesions):
            lid = cls_lesions[val_idx]
            test_lesions.add(lid)
            test_img_count += len(df[df['lesion_id'] == lid])
            val_idx += 1
            
        print(f"Class '{cls}': allocated {val_img_count} validation images, {test_img_count} test images.")
        
    # Split dataframe rows
    df_val_raw = df[df['lesion_id'].isin(val_lesions)]
    df_test_raw = df[df['lesion_id'].isin(test_lesions)]
    df_train_raw = df[~df['lesion_id'].isin(val_lesions) & ~df['lesion_id'].isin(test_lesions)]
    
    # We must ensure validation and test splits are EXACTLY balanced.
    val_sampled = []
    test_sampled = []
    
    for cls in classes:
        df_cls_val = df_val_raw[df_val_raw['dx'] == cls]
        df_cls_test = df_test_raw[df_test_raw['dx'] == cls]
        
        val_sampled.append(df_cls_val.sample(n=30, random_state=42))
        test_sampled.append(df_cls_test.sample(n=30, random_state=42))
        
    df_val = pd.concat(val_sampled).reset_index(drop=True)
    df_test = pd.concat(test_sampled).reset_index(drop=True)
    
    # The training set contains everything else
    used_image_ids = set(df_val['image_id']).union(set(df_test['image_id']))
    df_train = df_train_raw[~df_train_raw['image_id'].isin(used_image_ids)].reset_index(drop=True)
    
    print("\nFinal Split Distributions (Images):")
    print(f"Train size: {len(df_train)}")
    print(df_train['dx'].value_counts())
    
    print(f"\nVal size: {len(df_val)}")
    print(df_val['dx'].value_counts())
    
    print(f"\nTest size: {len(df_test)}")
    print(df_test['dx'].value_counts())
    
    # Save splits
    os.makedirs("processed", exist_ok=True)
    df_train.to_csv("processed/skin_train.csv", index=False)
    df_val.to_csv("processed/skin_val.csv", index=False)
    df_test.to_csv("processed/skin_test.csv", index=False)
    
    # Save classes list
    joblib_dir = "models/skin_cancer"
    os.makedirs(joblib_dir, exist_ok=True)
    joblib.dump(classes, os.path.join(joblib_dir, "classes.joblib"))
    joblib.dump(dx_to_idx, os.path.join(joblib_dir, "dx_to_idx.joblib"))
    
    print("\nPreprocessing complete. Saved splits to processed/ and class mappings to models/skin_cancer/.")

if __name__ == "__main__":
    preprocess_skin()
