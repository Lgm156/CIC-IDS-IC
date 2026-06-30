import os
import pandas as pd
import numpy as np
import joblib
import torch
from models import lstm

def main():
    print("Training LSTM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    os.makedirs("models/lstm", exist_ok=True)
    
    # Load data
    df_train = pd.read_parquet("processed/train_upper_stress.parquet")
    df_val = pd.read_parquet("processed/val.parquet")
    df_test = pd.read_parquet("processed/test.parquet")
    scaler = joblib.load("models/scaler.joblib")
    features = joblib.load("models/features.joblib")
    le = joblib.load("models/label_encoder.joblib")
    
    X_train = scaler.transform(df_train[features])
    y_train = df_train['LabelIdx'].values
    
    X_val = scaler.transform(df_val[features])
    y_val = df_val['LabelIdx'].values
    
    X_test = scaler.transform(df_test[features])
    y_test = df_test['LabelIdx'].values
    
    # Train
    model = lstm.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val, 
        num_classes=len(le.classes_),
        device=device,
        epochs=20,
        batch_size=1024,
        patience=5
    )
    
    # Save Model
    lstm.save_model(model, "models/lstm/model.pt")
    print("Saved model to models/lstm/model.pt")
    
    # Predict probabilities
    val_probs = lstm.predict_proba(model, X_val)
    test_probs = lstm.predict_proba(model, X_test)
    
    # Save probabilities & targets
    np.save("models/lstm/validation_posteriors.npy", val_probs)
    np.save("models/lstm/test_posteriors.npy", test_probs)
    np.save("models/lstm/validation_targets.npy", y_val)
    np.save("models/lstm/test_targets.npy", y_test)
    print("Saved validation and test posteriors.")

if __name__ == "__main__":
    main()
