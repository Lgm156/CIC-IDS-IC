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
    
    print("\nFinal Split Distributions:")
    print("Train:\n", df_train['TargetLabel'].value_counts())
    print("Val:\n", df_val['TargetLabel'].value_counts())
    print("Test:\n", df_test['TargetLabel'].value_counts())
    
    # Imbalance Ratio
    train_counts = df_train['TargetLabel'].value_counts()
    ir = train_counts.max() / train_counts.min()
    print(f"\nTraining Imbalance Ratio: {ir:.2f}")
    
    # Feature Scaling
    # Fit only on training data
    features = X.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df_train[features])
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(features, "models/features.joblib")
    
    # Save splits
    os.makedirs("processed", exist_ok=True)
    df_train.to_parquet("processed/train.parquet")
    df_val.to_parquet("processed/val.parquet")
    df_test.to_parquet("processed/test.parquet")
    
    # Save stats
    stats = pd.DataFrame({
        'Train': df_train['TargetLabel'].value_counts(),
        'Val': df_val['TargetLabel'].value_counts(),
        'Test': df_test['TargetLabel'].value_counts()
    })
    stats.to_csv("results/split_statistics.csv")
    print("\nSaved split statistics to results/split_statistics.csv")

if __name__ == "__main__":
    create_splits()
