import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import joblib
from models.skin_resnet import get_resnet18

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

def save_posteriors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load splits
    val_df = pd.read_csv("processed/skin_val.csv")
    test_df = pd.read_csv("processed/skin_test.csv")
    train_df = pd.read_csv("processed/skin_train.csv")
    
    # Define transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets & Dataloaders
    val_dataset = SkinDataset(val_df, transform=val_transform)
    test_dataset = SkinDataset(test_df, transform=val_transform)
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    # Load model
    model = get_resnet18(num_classes=7, pretrained=False).to(device)
    model.load_state_dict(torch.load("models/skin_cancer/best_resnet18.pt", map_location=device))
    model.eval()
    
    def extract(loader):
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.numpy())
                
        return np.concatenate(all_probs), np.concatenate(all_targets)
        
    print("Extracting validation posteriors...")
    val_probs, val_targets = extract(val_loader)
    
    print("Extracting test posteriors...")
    test_probs, test_targets = extract(test_loader)
    
    # Save posteriors and targets
    os.makedirs("results/skin_cancer", exist_ok=True)
    np.save("results/skin_cancer/validation_posteriors.npy", val_probs)
    np.save("results/skin_cancer/validation_targets.npy", val_targets)
    np.save("results/skin_cancer/test_posteriors.npy", test_probs)
    np.save("results/skin_cancer/test_targets.npy", test_targets)
    
    # Compute and save source prior P_s(y) from training set
    # P_s(y) must be sorted by label index
    counts = train_df['dx_idx'].value_counts().sort_index()
    ps = counts / counts.sum()
    ps_arr = ps.values
    np.save("results/skin_cancer/source_prior.npy", ps_arr)
    
    print("\nSource prior P_s(y):", ps_arr)
    print("Posteriors and source prior saved successfully.")

if __name__ == "__main__":
    save_posteriors()
