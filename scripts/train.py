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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df_train = pd.read_parquet("processed/train_upper_stress.parquet")
    df_val = pd.read_parquet("processed/val.parquet")
    
    scaler = joblib.load("models/scaler.joblib")
    features = joblib.load("models/features.joblib")
    
    X_train = scaler.transform(df_train[features])
    y_train = df_train['LabelIdx'].values
    
    X_val = scaler.transform(df_val[features])
    y_val = df_val['LabelIdx'].values
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Model
    model = MLP(input_dim=len(features), num_classes=len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training Loop
    epochs = 20
    patience = 5
    best_val_loss = float('inf')
    counter = 0
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        history.append({'epoch': epoch+1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_acc': val_acc})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break
                
    pd.DataFrame(history).to_csv("results/training_history.csv", index=False)

if __name__ == "__main__":
    train()
