import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import joblib
from models.mlp import MLP

def save_posteriors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load scaler, features, and label encoder
    scaler = joblib.load("models/forest/scaler.joblib")
    features = joblib.load("models/forest/features.joblib")
    le = joblib.load("models/forest/label_encoder.joblib")
    
    # Load best trained MLP model
    model = MLP(input_dim=len(features), num_classes=len(le.classes_)).to(device)
    model_path = "models/forest/mlp/best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Helper to calculate and extract posteriors
    def extract_probs(file_path):
        df = pd.read_parquet(file_path)
        X = scaler.transform(df[features])
        y = df['LabelIdx'].values
        
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(batch_y.numpy())
                
        return np.concatenate(all_probs), np.concatenate(all_targets)

    print("Extracting validation posteriors...")
    val_probs, val_targets = extract_probs("processed/forest/val.parquet")
    
    print("Extracting test posteriors...")
    test_probs, test_targets = extract_probs("processed/forest/test.parquet")
    
    # Save validation & test posteriors and targets
    model_dir = "models/forest/mlp"
    os.makedirs(model_dir, exist_ok=True)
    
    np.save(os.path.join(model_dir, "validation_posteriors.npy"), val_probs)
    np.save(os.path.join(model_dir, "validation_targets.npy"), val_targets)
    np.save(os.path.join(model_dir, "test_posteriors.npy"), test_probs)
    np.save(os.path.join(model_dir, "test_targets.npy"), test_targets)
    print(f"Saved posteriors and targets to {model_dir}/")
    
    # Compute and save training class prior P_s(y)
    df_train = pd.read_parquet("processed/forest/train.parquet")
    counts = df_train['LabelIdx'].value_counts().sort_index()
    ps = counts / counts.sum()
    ps_arr = ps.values
    
    results_dir = "results/forest"
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "source_prior.npy"), ps_arr)
    print(f"Saved training set source prior to {results_dir}/source_prior.npy. Prior: {ps_arr}")

if __name__ == "__main__":
    save_posteriors()
