import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, num_features) or (batch, 1, num_features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # last time step
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

def train(X_train, y_train, X_val=None, y_val=None, **kwargs):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    input_dim = X_train.shape[1]
    # Handle potentially offline classes or map based on unique classes
    num_classes = kwargs.get('num_classes', len(np.unique(y_train)))
    
    hidden_dim = kwargs.get('hidden_dim', 128)
    num_layers = kwargs.get('num_layers', 1)
    dropout = kwargs.get('dropout', 0.3)
    lr = kwargs.get('lr', 1e-3)
    batch_size = kwargs.get('batch_size', 1024)
    epochs = kwargs.get('epochs', 20)
    patience = kwargs.get('patience', 5)
    device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = LSTMClassifier(input_dim, hidden_dim, num_classes, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    has_val = X_val is not None and y_val is not None
    if has_val:
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        if has_val:
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == batch_y).item()
                    
            val_loss /= len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
            
    if has_val:
        model.load_state_dict(best_model_wts)
        
    return model

def predict_proba(model, X):
    model.eval()
    device = next(model.parameters()).device
    if isinstance(X, np.ndarray):
        X_tensor = torch.FloatTensor(X)
    else:
        X_tensor = X
        
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for batch_X, in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs, axis=0)

def predict(model, X):
    probs = predict_proba(model, X)
    return np.argmax(probs, axis=1)

def save_model(model, path):
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': model.lstm.input_size,
        'hidden_dim': model.lstm.hidden_size,
        'num_classes': model.fc.out_features,
        'num_layers': model.lstm.num_layers,
        'dropout': model.dropout.p
    }, path)

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    model = LSTMClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model
