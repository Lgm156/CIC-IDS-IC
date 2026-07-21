import pandas as pd
import os

def analyze_skin_dataset():
    metadata_path = "SkinCancerDataset/HAM10000_metadata.csv"
    print(f"Loading metadata from {metadata_path}...")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    print(f"Metadata shape: {df.shape}")
    
    label_col = 'dx'
    if label_col not in df.columns:
        print(f"Error: '{label_col}' column not found. Available columns: {df.columns.tolist()}")
        return

    original_counts = df[label_col].value_counts()
    print("\nOriginal Label Counts:")
    print(original_counts)
    
    os.makedirs("results", exist_ok=True)
    original_counts.to_csv("results/skin_original_class_distribution.csv")
    
    # In HAM10000, we map all 7 classes as they are all kept (no classes dropped or merged)
    def map_label(label):
        l = str(label).lower().strip()
        # The valid labels are: nv, mel, bkl, bcc, akiec, vasc, df
        valid_labels = {'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'}
        if l in valid_labels:
            return l
        return None # Drop if there is anything else

    df['TargetLabel'] = df[label_col].apply(map_label)
    
    # Drop samples with None TargetLabel (if any)
    df = df.dropna(subset=['TargetLabel'])
    
    mapped_counts = df['TargetLabel'].value_counts()
    print("\nMapped Label Counts (7 Classes):")
    print(mapped_counts)
    
    mapped_counts.to_csv("results/skin_final_class_distribution.csv")
    
    # Verification
    expected_classes = {'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'}
    actual_classes = set(mapped_counts.index)
    
    if not expected_classes.issubset(actual_classes):
        missing = expected_classes - actual_classes
        print(f"\n!!! ERROR: Missing classes: {missing}")
        return
    
    print("\nVerification passed. 7 classes present. Ready for preprocessing.")

if __name__ == "__main__":
    analyze_skin_dataset()
