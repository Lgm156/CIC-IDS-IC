import pandas as pd
import os
import glob
from pathlib import Path

def analyze_dataset():
    data_dir = "CICIDS"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files.")
    
    dfs = []
    for f in csv_files:
        print(f"Loading {f}...")
        # CICIDS CSVs often have encoding issues or different delimiters, 
        # but usually they are standard CSV with latin-1 or utf-8.
        try:
            df = pd.read_csv(f, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='utf-8')
        dfs.append(df)
        
    print("Merging dataframes...")
    full_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Full dataframe shape: {full_df.shape}")
    
    # Clean column names (strip whitespace)
    full_df.columns = full_df.columns.str.strip()
    
    label_col = 'Label'
    if label_col not in full_df.columns:
        print(f"Error: '{label_col}' column not found. Available columns: {full_df.columns.tolist()}")
        return

    original_counts = full_df[label_col].value_counts()
    print("\nOriginal Label Counts:")
    print(original_counts)
    
    os.makedirs("results", exist_ok=True)
    original_counts.to_csv("results/original_class_distribution.csv")
    
    # Class Mapping (Updated for 7 classes)
    def map_label(label):
        l = str(label).lower().strip()
        if 'benign' in l: return 'BENIGN'
        if 'ddos' in l: return 'DDoS'
        if 'portscan' in l: return 'PortScan'
        if 'bot' in l: return 'Bot'
        if 'ftp-patator' in l or 'ssh-patator' in l: return 'BruteForce'
        if 'dos' in l: return 'DoS'
        if 'web attack' in l: return 'WebAttack'
        return None # Drop others

    full_df['TargetLabel'] = full_df[label_col].apply(map_label)
    
    # Drop samples with None TargetLabel (Infiltration, Heartbleed, etc.)
    full_df = full_df.dropna(subset=['TargetLabel'])
    
    mapped_counts = full_df['TargetLabel'].value_counts()
    print("\nMapped Label Counts (7 Classes):")
    print(mapped_counts)
    
    mapped_counts.to_csv("results/final_class_distribution.csv")
    
    # Verification
    expected_classes = {'BENIGN', 'DDoS', 'PortScan', 'DoS', 'Bot', 'BruteForce', 'WebAttack'}
    actual_classes = set(mapped_counts.index)
    
    if not expected_classes.issubset(actual_classes):
        missing = expected_classes - actual_classes
        print(f"\n!!! ERROR: Missing classes: {missing}")
        return
    
    print("\nVerification passed. 7 classes present. Ready for preprocessing.")
    
    # Save the dataframe temporarily for the next script to avoid re-merging if possible
    # or just let preprocess.py handle it. The plan says preprocess.py handles merge.
    # I'll update preprocess.py to use this robust mapping.

if __name__ == "__main__":
    analyze_dataset()
