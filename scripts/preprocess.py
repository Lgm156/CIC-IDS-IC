import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess():
    data_dir = "CICIDS"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='utf-8')
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # Drop Unwanted Columns early
    cols_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Robust Label Mapping
    def map_label(label):
        l = str(label).lower().strip()
        if 'benign' in l: return 'BENIGN'
        if 'ddos' in l: return 'DDoS'
        if 'portscan' in l: return 'PortScan'
        if 'bot' in l: return 'Bot'
        if 'ftp-patator' in l or 'ssh-patator' in l: return 'BruteForce'
        if 'dos' in l: return 'DoS'
        if 'web attack' in l: return 'WebAttack'
        return None

    df['TargetLabel'] = df['Label'].apply(map_label)
    df = df.dropna(subset=['TargetLabel'])
    
    print(f"Cleaning data... Initial shape: {df.shape}")
    
    # Replace inf with NaN then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    print(f"Post-cleaning shape: {df.shape}")
    
    os.makedirs("processed", exist_ok=True)
    df.to_parquet("processed/processed_dataset.parquet")
    print("Saved processed/processed_dataset.parquet")

if __name__ == "__main__":
    preprocess()
