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
    
    scaler = joblib.load("models/scaler.joblib")
    features = joblib.load("models/features.joblib")
    le = joblib.load("models/label_encoder.joblib")
    
    model = MLP(input_dim=len(features), num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    
    def get_posteriors(file_path):
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

    print("Saving validation posteriors...")
    val_probs, val_targets = get_posteriors("processed/val.parquet")
    np.save("results/validation_posteriors.npy", val_probs)
    np.save("results/validation_targets.npy", val_targets)
    
    print("Saving test posteriors...")
    test_probs, test_targets = get_posteriors("processed/test.parquet")
    np.save("results/test_posteriors.npy", test_probs)
    np.save("results/test_targets.npy", test_targets)
    
    # Save Source Prior Ps(y)
    df_train = pd.read_parquet("processed/train.parquet")
    ps = df_train['LabelIdx'].value_counts(normalize=True).sort_index().values
    np.save("results/source_prior.npy", ps)
    print("Saved source prior Ps(y)")

if __name__ == "__main__":
    save_posteriors()
