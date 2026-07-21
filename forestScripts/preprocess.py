import pandas as pd
import numpy as np
import os

def preprocess():
    csv_file = "forestCover/covtype.csv"
    print(f"Loading forest cover dataset from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"Error: Dataset file not found at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded dataset. Shape: {df.shape}")
    
    df.columns = df.columns.str.strip()
    
    # Class mapping for Forest Cover Type
    cover_type_mapping = {
        1: 'Spruce_Fir',
        2: 'Lodgepole_Pine',
        3: 'Ponderosa_Pine',
        4: 'Cottonwood_Willow',
        5: 'Aspen',
        6: 'Douglas_fir',
        7: 'Krummholz'
    }
    
    # Add TargetLabel
    df['TargetLabel'] = df['Cover_Type'].map(cover_type_mapping)
    df = df.dropna(subset=['TargetLabel'])
    
    print(f"Cleaning data... Initial shape: {df.shape}")
    
    # Replace inf with NaN then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Drop duplicates
    df = df.drop_duplicates()
    print(f"Post-cleaning shape (after dropping duplicates): {df.shape}")
    
    os.makedirs("processed/forest", exist_ok=True)
    output_path = "processed/forest/processed_dataset.parquet"
    
    print("Saving processed dataset to parquet...")
    df.to_parquet(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    preprocess()
