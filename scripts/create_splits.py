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
    
    # Helper to build train splits with oversampling support
    def build_train_split(df_train_full, target_counts, use_replacement=False):
        dfs = []
        for label, target_count in target_counts.items():
            df_label = df_train_full[df_train_full['TargetLabel'] == label]
            available_count = len(df_label)
            if target_count > available_count:
                if use_replacement:
                    df_sampled = df_label.sample(n=target_count, replace=True, random_state=42)
                else:
                    df_sampled = df_label
            else:
                df_sampled = df_label.sample(n=target_count, replace=False, random_state=42)
            dfs.append(df_sampled)
        return pd.concat(dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 1. Baseline Target Counts
    target_train_counts_baseline = {
        'BENIGN': 100000,
        'DoS': 50000,
        'PortScan': 25000,
        'DDoS': 12000,
        'BruteForce': 6000,
        'WebAttack': 1000,
        'Bot': 700
    }

    # 2. Upper Stress Target Counts (+50% shift delta of absolute extremes)
    target_train_counts_upper = {
        'BENIGN': 49192,
        'DoS': 32767,
        'DDoS': 20294,
        'BruteForce': 18317,
        'WebAttack': 16681,
        'PortScan': 16664,
        'Bot': 16494
    }

    # 3. Lower Stress Target Counts (-50% shift delta of absolute extremes)
    target_train_counts_lower = {
        'BENIGN': 166848,
        'DoS': 716,
        'DDoS': 562,
        'BruteForce': 562,
        'WebAttack': 562,
        'PortScan': 562,
        'Bot': 562
    }

    print("\nBuilding training splits...")
    df_train = build_train_split(df_train_full, target_train_counts_baseline, use_replacement=False)
    df_train_upper = build_train_split(df_train_full, target_train_counts_upper, use_replacement=False)
    df_train_lower = build_train_split(df_train_full, target_train_counts_lower, use_replacement=False)

    print("\nFinal Split Distributions:")
    print("Train (Baseline):\n", df_train['TargetLabel'].value_counts())
    print("Train (Upper Stress):\n", df_train_upper['TargetLabel'].value_counts())
    print("Train (Lower Stress):\n", df_train_lower['TargetLabel'].value_counts())
    print("Val:\n", df_val['TargetLabel'].value_counts())
    print("Test:\n", df_test['TargetLabel'].value_counts())
    
    # Imbalance Ratios
    for name, df_t in [("Baseline", df_train), ("Upper Stress", df_train_upper), ("Lower Stress", df_train_lower)]:
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
    df_val.to_parquet("processed/val.parquet")
    df_test.to_parquet("processed/test.parquet")
    
    # Save stats
    stats = pd.DataFrame({
        'Train_Baseline': df_train['TargetLabel'].value_counts(),
        'Train_Upper_Stress': df_train_upper['TargetLabel'].value_counts(),
        'Train_Lower_Stress': df_train_lower['TargetLabel'].value_counts(),
        'Val': df_val['TargetLabel'].value_counts(),
        'Test': df_test['TargetLabel'].value_counts()
    })
    stats.to_csv("results/split_statistics_stress.csv")
    stats.to_csv("results/split_statistics.csv") # backward compatibility
    print("\nSaved split statistics to results/split_statistics_stress.csv")

if __name__ == "__main__":
    create_splits()
