import pandas as pd
import os

def analyze_dataset():
    csv_file = "forestCover/covtype.csv"
    print(f"Loading forest cover dataset from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"Error: Dataset file not found at {csv_file}")
        return
        
    # Read the dataset (only the label column first to speed up if needed, 
    # but we need the whole dataset eventually. Since we are just analyzing 
    # the labels here, we can read the entire file or just the column.
    # Reading the label column is much faster for a 75MB file, but let's read the Cover_Type column)
    try:
        df_labels = pd.read_csv(csv_file, usecols=['Cover_Type'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Loaded label column. Total rows: {len(df_labels)}")
    
    label_col = 'Cover_Type'
    original_counts = df_labels[label_col].value_counts().sort_index()
    print("\nOriginal Cover_Type Counts:")
    for cover_type, count in original_counts.items():
        print(f"Cover_Type {cover_type}: {count}")
        
    os.makedirs("results/forest", exist_ok=True)
    original_counts.to_csv("results/forest/original_class_distribution.csv")
    
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
    
    df_labels['TargetLabel'] = df_labels[label_col].map(cover_type_mapping)
    
    mapped_counts = df_labels['TargetLabel'].value_counts()
    print("\nMapped Label Counts (7 Classes):")
    print(mapped_counts)
    
    mapped_counts.to_csv("results/forest/final_class_distribution.csv")
    
    # Verification
    expected_classes = set(cover_type_mapping.values())
    actual_classes = set(mapped_counts.index)
    
    if not expected_classes.issubset(actual_classes):
        missing = expected_classes - actual_classes
        print(f"\n!!! ERROR: Missing classes: {missing}")
        return
        
    print("\nVerification passed. All 7 cover type classes are present. Ready for preprocessing.")

if __name__ == "__main__":
    analyze_dataset()
