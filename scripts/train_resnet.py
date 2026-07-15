import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import joblib
from models.skin_resnet import get_resnet18
from tqdm import tqdm

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['dx_idx']
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load splits
    train_df = pd.read_csv("processed/skin_train.csv")
    val_df = pd.read_csv("processed/skin_val.csv")
    
    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets & Dataloaders
    train_dataset = SkinDataset(train_df, transform=train_transform)
    val_dataset = SkinDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = get_resnet18(num_classes=7, pretrained=True).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # Fine-tuning: smaller learning rate is better
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    epochs = 8
    best_val_acc = 0.0
    history = []
    
    os.makedirs("models/skin_cancer", exist_ok=True)
    os.makedirs("results/skin_cancer", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            # Update tqdm progress bar
            loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Train Acc = {epoch_train_acc:.4f} | "
              f"Val Loss = {epoch_val_loss:.4f}, Val Acc = {epoch_val_acc:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "models/skin_cancer/best_resnet18.pt")
            print(f"--> Saved new best model with Val Acc = {best_val_acc:.4f}")
            
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv("results/skin_cancer/training_history.csv", index=False)
    print("Training complete. History saved.")

if __name__ == "__main__":
    train()
