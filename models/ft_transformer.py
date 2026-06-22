import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        # Each numerical feature has its own weight and bias vector of size d_token
        self.weights = nn.Parameter(torch.randn(num_features, d_token))
        self.biases = nn.Parameter(torch.randn(num_features, d_token))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        # x shape: (batch, num_features)
        # weight shape: (num_features, d_token)
        # bias shape: (num_features, d_token)
        # Output shape: (batch, num_features, d_token)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)

class FTTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_token=64, n_blocks=2, n_heads=4, d_ffn=128, dropout=0.1):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        nn.init.xavier_uniform_(self.cls_token)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        self.ln = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, num_classes)
        
    def forward(self, x):
        # x shape: (batch, num_features)
        tokens = self.tokenizer(x) # (batch, num_features, d_token)
        
        # Prepend CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch, 1, d_token)
        tokens = torch.cat([cls_tokens, tokens], dim=1) # (batch, num_features + 1, d_token)
        
        # Pass through Transformer
        out = self.transformer(tokens) # (batch, num_features + 1, d_token)
        
        # Get CLS token output
        cls_out = out[:, 0, :] # (batch, d_token)
        cls_out = self.ln(cls_out)
        logits = self.head(cls_out)
        return logits

def train(X_train, y_train, X_val=None, y_val=None, **kwargs):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    input_dim = X_train.shape[1]
    num_classes = kwargs.get('num_classes', len(np.unique(y_train)))
    
    d_token = kwargs.get('d_token', 64)
    n_blocks = kwargs.get('n_blocks', 2)
    n_heads = kwargs.get('n_heads', 4)
    d_ffn = kwargs.get('d_ffn', 128)
    dropout = kwargs.get('dropout', 0.1)
    
    lr = kwargs.get('lr', 1e-3)
    batch_size = kwargs.get('batch_size', 1024)
    epochs = kwargs.get('epochs', 20)
    patience = kwargs.get('patience', 5)
    device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = FTTransformer(
        num_features=input_dim,
        num_classes=num_classes,
        d_token=d_token,
        n_blocks=n_blocks,
        n_heads=n_heads,
        d_ffn=d_ffn,
        dropout=dropout
    ).to(device)
    
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
        'num_features': model.tokenizer.weights.shape[0],
        'num_classes': model.head.out_features,
        'd_token': model.tokenizer.weights.shape[1],
        'n_blocks': len(model.transformer.layers),
        'n_heads': model.transformer.layers[0].self_attn.num_heads,
        'd_ffn': model.transformer.layers[0].linear1.out_features,
        'dropout': model.transformer.layers[0].dropout.p
    }, path)

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    model = FTTransformer(
        num_features=checkpoint['num_features'],
        num_classes=checkpoint['num_classes'],
        d_token=checkpoint['d_token'],
        n_blocks=checkpoint['n_blocks'],
        n_heads=checkpoint['n_heads'],
        d_ffn=checkpoint['d_ffn'],
        dropout=checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model
