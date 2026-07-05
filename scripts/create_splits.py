import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def create_splits():
    df = pd.read_parquet("processed/processed_dataset.parquet")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Label Encoding
    le = LabelEncoder()
    df['LabelIdx'] = le.fit_transform(df['TargetLabel'])
    joblib.dump(le, "models/label_encoder.joblib")
    print(f"Classes: {le.classes_}")
    
    # Feature/Target split
    X = df.drop(columns=['Label', 'TargetLabel', 'LabelIdx'])
    y = df['LabelIdx']
    
    # We need balanced Val and Test sets. 
    # Smallest class (Bot) has 1966 samples after cleaning? Let's check.
    counts = df['TargetLabel'].value_counts()
    print("\nCounts after cleaning:")
    print(counts)
    
    # Target 500 samples per class for Val and Test (total 1000 per class reserved)
    # Smallest class is Bot (approx 1900). 500 is safe.
    n_samples_per_class = 500
    
    val_indices = []
    test_indices = []
    train_indices = []
    
    for label in le.classes_:
        idx = df[df['TargetLabel'] == label].index.tolist()
        np.random.seed(42)
        np.random.shuffle(idx)
        
        val_indices.extend(idx[:n_samples_per_class])
        test_indices.extend(idx[n_samples_per_class:2*n_samples_per_class])
        train_indices.extend(idx[2*n_samples_per_class:])
        
    df_val = df.loc[val_indices]
    df_test = df.loc[test_indices]
    df_train_full = df.loc[train_indices]
    
    # Now create Long-Tailed Training Set
    # Target distribution (approx 100-150 ratio):
    # BENIGN: 100000
    # DoS: 50000
    # PortScan: 25000
    # DDoS: 12000
    # BruteForce: 6000
    # WebAttack: 1000
    # Bot: 700
    
    target_train_counts = {
        'BENIGN': 150000,
        'DoS': 50000,
        'PortScan': 25000,
        'DDoS': 12000,
        'BruteForce': 6000,
        'WebAttack': 1000,
        'Bot': 700
    }
    
    final_train_indices = []
    for label, count in target_train_counts.items():
        idx = df_train_full[df_train_full['TargetLabel'] == label].index.tolist()
        np.random.seed(42)
        np.random.shuffle(idx)
        final_train_indices.extend(idx[:count])
        
    df_train = df.loc[final_train_indices]
    
    # Extreme‑stress: halve the minority‑to‑majority ratio
    # Compute baseline class counts
    N_max = train_counts.max()
    N_min = train_counts.min()
    # Compute baseline ratio and new target ratio
    r = N_min / N_max if N_max != 0 else 0
    delta = r / 2.0
    r_extreme = r - delta  # equivalent to r / 2
    N_min_new = int(round(r_extreme * N_max))
    if N_min_new < 1:
        print(f"[WARNING] Computed N_min_new={N_min_new} < 1; clipping to 1.")
        N_min_new = 1
    # Build target counts: keep original for all classes, adjust minority class(es)
    target_train_counts_extreme = train_counts.to_dict()
    minority_classes = train_counts[train_counts == N_min].index.tolist()
    for cls in minority_classes:
        target_train_counts_extreme[cls] = N_min_new
    # Note: undersampling without replacement, no replication needed

    print("\nBuilding training splits...")
    df_train = build_train_split(df_train_full, target_train_counts_baseline, use_replacement=True)
    df_train_upper = build_train_split(df_train_full, target_train_counts_upper, use_replacement=True)
    df_train_lower = build_train_split(df_train_full, target_train_counts_lower, use_replacement=True)
    df_train_extreme = build_train_split(df_train_full, target_train_counts_extreme, use_replacement=False)

    print("\nFinal Split Distributions:")
    print("Train (Baseline):\n", df_train['TargetLabel'].value_counts())
    print("Train (Upper Stress):\n", df_train_upper['TargetLabel'].value_counts())
    # Verification table: baseline vs extreme stress counts
    print("\nVerification Table (Baseline vs Extreme Stress):")
    verification_df = pd.DataFrame({
        "Class": train_counts.index,
        "Baseline": train_counts.values,
        "Extreme_Stress": [target_train_counts_extreme[cls] for cls in train_counts.index]
    })
    print(verification_df.to_string(index=False))
    # Ratios summary
    achieved_ratio = min(verification_df["Extreme_Stress"]) / max(verification_df["Extreme_Stress"]) if max(verification_df["Extreme_Stress"]) > 0 else None
    baseline_ratio = max_count / min_count if min_count != 0 else None
    print(f"ratio_baseline (max/min): {baseline_ratio:.4f}" if baseline_ratio is not None else "ratio_baseline: undefined")
    print(f"achieved_ratio (Extreme): {achieved_ratio:.4f}")
    print(f"Total samples - Baseline: {train_counts.sum()}, Extreme: {verification_df['Extreme_Stress'].sum()}")
    print("Val:\n", df_val['TargetLabel'].value_counts())
    print("Test:\n", df_test['TargetLabel'].value_counts())
    
    # Imbalance Ratios
    for name, df_t in [("Baseline", df_train), ("Upper Stress", df_train_upper), ("Lower Stress", df_train_lower), ("Extreme Stress", df_train_extreme)]:
        counts_t = df_t['TargetLabel'].value_counts()
        ir = counts_t.max() / counts_t.min()
        print(f"{name} Training Imbalance Ratio: {ir:.2f}")
    
    # Feature Scaling
    # Fit only on baseline training data
    features = X.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df_train[features])
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(features, "models/features.joblib")
    
    # Save splits
    os.makedirs("processed", exist_ok=True)
    df_train.to_parquet("processed/train.parquet")
    df_train_upper.to_parquet("processed/train_upper_stress.parquet")
    df_train_lower.to_parquet("processed/train_lower_stress.parquet")
    df_train_extreme.to_parquet("processed/train_extreme_stress.parquet")
    df_val.to_parquet("processed/val.parquet")
    df_test.to_parquet("processed/test.parquet")
    
    # Save stats
    stats = pd.DataFrame({
        'Train_Baseline': df_train['TargetLabel'].value_counts(),
        'Train_Upper_Stress': df_train_upper['TargetLabel'].value_counts(),
        'Train_Lower_Stress': df_train_lower['TargetLabel'].value_counts(),
        'Train_Extreme_Stress': df_train_extreme['TargetLabel'].value_counts(),
        'Val': df_val['TargetLabel'].value_counts(),
        'Test': df_test['TargetLabel'].value_counts()
    })
    stats.to_csv("results/split_statistics_stress.csv")
    stats.to_csv("results/split_statistics.csv") # backward compatibility
    print("\nSaved split statistics to results/split_statistics_stress.csv")

if __name__ == "__main__":
    create_splits()
