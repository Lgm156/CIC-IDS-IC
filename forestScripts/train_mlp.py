import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import joblib
from models.mlp import MLP
from tqdm import tqdm

def main():
    # Set seeds for strict reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print("Training MLP (Neural Network) on Forest Cover Dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    model_dir = "models/forest/mlp"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("results/forest", exist_ok=True)
    
    # Load data splits
    df_train = pd.read_parquet("processed/forest/train.parquet")
    df_val = pd.read_parquet("processed/forest/val.parquet")
    df_test = pd.read_parquet("processed/forest/test.parquet")
    
    # Load scaler, features, and label encoder
    scaler = joblib.load("models/forest/scaler.joblib")
    features = joblib.load("models/forest/features.joblib")
    le = joblib.load("models/forest/label_encoder.joblib")
    
    # Scale features
    X_train = scaler.transform(df_train[features])
    y_train = df_train['LabelIdx'].values
    
    X_val = scaler.transform(df_val[features])
    y_val = df_val['LabelIdx'].values
    
    X_test = scaler.transform(df_test[features])
    y_test = df_test['LabelIdx'].values
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Train loader (balanced shuffle)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    # Initialize Model
    model = MLP(input_dim=len(features), num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training Loop parameters
    epochs = 20
    patience = 5
    best_val_loss = float('inf')
    counter = 0
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_X, batch_y in loop:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total_train += batch_y.size(0)
            correct_train += predicted.eq(batch_y).sum().item()
            
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total_val += batch_y.size(0)
                correct_val += predicted.eq(batch_y).sum().item()
                
        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        history.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"--> Saved best model checkpoint to {model_dir}/best_model.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    # Save training history
    pd.DataFrame(history).to_csv("results/forest/training_history_mlp.csv", index=False)
    
    # Load best weights for inference
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
    model.eval()
    
    # Predict posteriors on Val and Test
    def get_posteriors(X_data, y_data):
        dataset = TensorDataset(torch.FloatTensor(X_data), torch.LongTensor(y_data))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs)
        
    print("\nExtracting posteriors...")
    val_probs = get_posteriors(X_val, y_val)
    test_probs = get_posteriors(X_test, y_test)
    
    # Save posteriors and targets
    np.save(os.path.join(model_dir, "validation_posteriors.npy"), val_probs)
    np.save(os.path.join(model_dir, "test_posteriors.npy"), test_probs)
    np.save(os.path.join(model_dir, "validation_targets.npy"), y_val)
    np.save(os.path.join(model_dir, "test_targets.npy"), y_test)
    print("Saved validation and test posteriors.")
    
    # Save training set class prior Ps(y)
    # Counts of class indices sorted
    counts = df_train['LabelIdx'].value_counts().sort_index()
    ps = counts / counts.sum()
    ps_arr = ps.values
    np.save("results/forest/source_prior.npy", ps_arr)
    print(f"Saved source prior to results/forest/source_prior.npy. Prior: {ps_arr}")

if __name__ == "__main__":
    main()
