import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# =====================================================================
# CONFIGURATION SECTION - Easily tweak split counts here
# =====================================================================

# Number of samples per class for validation and test splits (must be balanced)
VAL_TEST_SAMPLES_PER_CLASS = 500

# Target training counts per class for baseline split
TARGET_TRAIN_COUNTS_BASELINE = {
    'Lodgepole_Pine': 150000,
    'Spruce_Fir': 100000,
    'Ponderosa_Pine': 20000,
    'Krummholz': 10000,
    'Douglas_fir': 8000,
    'Aspen': 4000,
    'Cottonwood_Willow': 1000
}

# Target training counts per class for upper stress split (higher volume)
TARGET_TRAIN_COUNTS_UPPER = {
    'Lodgepole_Pine': 180000,
    'Spruce_Fir': 120000,
    'Ponderosa_Pine': 25000,
    'Krummholz': 12000,
    'Douglas_fir': 10000,
    'Aspen': 5000,
    'Cottonwood_Willow': 1200
}

# Target training counts per class for lower stress split (lower volume)
TARGET_TRAIN_COUNTS_LOWER = {
    'Lodgepole_Pine': 120000,
    'Spruce_Fir': 80000,
    'Ponderosa_Pine': 15000,
    'Krummholz': 8000,
    'Douglas_fir': 6000,
    'Aspen': 3000,
    'Cottonwood_Willow': 800
}

# Target training counts per class for extreme stress split (extremely imbalanced)
TARGET_TRAIN_COUNTS_EXTREME = {
    'Lodgepole_Pine': 150000,
    'Spruce_Fir': 100000,
    'Ponderosa_Pine': 20000,
    'Krummholz': 10000,
    'Douglas_fir': 8000,
    'Aspen': 4000,
    'Cottonwood_Willow': 200  # Rarest class is heavily squeezed
}

# =====================================================================

def create_splits():
    processed_path = "processed/forest/processed_dataset.parquet"
    print(f"Loading dataset from {processed_path}...")
    
    if not os.path.exists(processed_path):
        print(f"Error: Processed dataset not found at {processed_path}")
        return
        
    df = pd.read_parquet(processed_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Label Encoding
    le = LabelEncoder()
    df['LabelIdx'] = le.fit_transform(df['TargetLabel'])
    
    models_dir = "models/forest"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
    print(f"Encoded classes: {le.classes_}")
    
    # We need balanced Val and Test sets.
    # Group indices by class to perform balanced splitting
    val_indices = []
    test_indices = []
    train_pool_indices = {}
    
    for label in le.classes_:
        # Find row indices for this class
        idx = df[df['TargetLabel'] == label].index.tolist()
        
        # Shuffle with a fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(idx)
        
        # Allocate validation and test sets
        val_indices.extend(idx[:VAL_TEST_SAMPLES_PER_CLASS])
        test_indices.extend(idx[VAL_TEST_SAMPLES_PER_CLASS : 2*VAL_TEST_SAMPLES_PER_CLASS])
        train_pool_indices[label] = idx[2*VAL_TEST_SAMPLES_PER_CLASS:]
        
    df_val = df.loc[val_indices].reset_index(drop=True)
    df_test = df.loc[test_indices].reset_index(drop=True)
    
    # Function to build training split from candidate pools
    def build_train_split(target_counts):
        final_indices = []
        np.random.seed(42)
        for label, count in target_counts.items():
            pool = train_pool_indices[label]
            if count > len(pool):
                print(f"[Warning] Requested {count} samples for {label}, but only {len(pool)} available. Sampling with replacement.")
                sampled = np.random.choice(pool, size=count, replace=True)
            else:
                sampled = np.random.choice(pool, size=count, replace=False)
            final_indices.extend(sampled)
        return df.loc[final_indices].reset_index(drop=True)

    print("\nBuilding training splits...")
    df_train = build_train_split(TARGET_TRAIN_COUNTS_BASELINE)
    df_train_upper = build_train_split(TARGET_TRAIN_COUNTS_UPPER)
    df_train_lower = build_train_split(TARGET_TRAIN_COUNTS_LOWER)
    df_train_extreme = build_train_split(TARGET_TRAIN_COUNTS_EXTREME)
    
    print("\nFinal Split Distributions (Images/Samples):")
    print(f"Val split size: {len(df_val)} (balanced: {VAL_TEST_SAMPLES_PER_CLASS} per class)")
    print(f"Test split size: {len(df_test)} (balanced: {VAL_TEST_SAMPLES_PER_CLASS} per class)")
    print(f"Train Baseline size: {len(df_train)}")
    print(df_train['TargetLabel'].value_counts())
    
    # Imbalance Ratios
    for name, df_t in [("Baseline", df_train), ("Upper Stress", df_train_upper), ("Lower Stress", df_train_lower), ("Extreme Stress", df_train_extreme)]:
        counts_t = df_t['TargetLabel'].value_counts()
        ir = counts_t.max() / counts_t.min()
        print(f"{name} Training Imbalance Ratio: {ir:.2f}")
        
    # Feature Scaling
    # Define features: all columns except metadata and labels
    exclude_cols = {'Cover_Type', 'TargetLabel', 'LabelIdx'}
    features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\nScaling features... Feature count: {len(features)}")
    scaler = StandardScaler()
    scaler.fit(df_train[features])
    
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(features, os.path.join(models_dir, "features.joblib"))
    print("Saved scaler.joblib and features.joblib.")
    
    # Save splits
    processed_dir = "processed/forest"
    df_train.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    df_train_upper.to_parquet(os.path.join(processed_dir, "train_upper_stress.parquet"), index=False)
    df_train_lower.to_parquet(os.path.join(processed_dir, "train_lower_stress.parquet"), index=False)
    df_train_extreme.to_parquet(os.path.join(processed_dir, "train_extreme_stress.parquet"), index=False)
    df_val.to_parquet(os.path.join(processed_dir, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)
    print("Saved split Parquet files.")
    
    # Save stats
    stats = pd.DataFrame({
        'Train_Baseline': df_train['TargetLabel'].value_counts(),
        'Train_Upper_Stress': df_train_upper['TargetLabel'].value_counts(),
        'Train_Lower_Stress': df_train_lower['TargetLabel'].value_counts(),
        'Train_Extreme_Stress': df_train_extreme['TargetLabel'].value_counts(),
        'Val': df_val['TargetLabel'].value_counts(),
        'Test': df_test['TargetLabel'].value_counts()
    })
    
    results_dir = "results/forest"
    os.makedirs(results_dir, exist_ok=True)
    stats.to_csv(os.path.join(results_dir, "split_statistics_stress.csv"))
    stats.to_csv(os.path.join(results_dir, "split_statistics.csv"))
    print("Saved split statistics to results/forest/split_statistics_stress.csv")

if __name__ == "__main__":
    create_splits()
