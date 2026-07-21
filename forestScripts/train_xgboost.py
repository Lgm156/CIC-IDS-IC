import os
import pandas as pd
import numpy as np
import joblib
from models import xgboost_model

def main():
    print("Training XGBoost on Forest Cover Dataset...")
    
    # Paths
    model_dir = "models/forest/xgboost"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data splits
    df_train = pd.read_parquet("processed/forest/train.parquet")
    df_val = pd.read_parquet("processed/forest/val.parquet")
    df_test = pd.read_parquet("processed/forest/test.parquet")
    
    # Load scaler, features, and label encoder
    scaler = joblib.load("models/forest/scaler.joblib")
    features = joblib.load("models/forest/features.joblib")
    le = joblib.load("models/forest/label_encoder.joblib")
    
    X_train = scaler.transform(df_train[features])
    y_train = df_train['LabelIdx'].values
    
    X_val = scaler.transform(df_val[features])
    y_val = df_val['LabelIdx'].values
    
    X_test = scaler.transform(df_test[features])
    y_test = df_test['LabelIdx'].values
    
    # Parameters (similar to original train_xgboost.py)
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    print(f"Training parameters: {params}")
    
    # Train
    model = xgboost_model.train(X_train, y_train, **params)
    
    # Save Model
    model_path = os.path.join(model_dir, "model.joblib")
    xgboost_model.save_model(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Predict probabilities (posteriors)
    val_probs = xgboost_model.predict_proba(model, X_val)
    test_probs = xgboost_model.predict_proba(model, X_test)
    
    # Save probabilities & targets for calibration
    np.save(os.path.join(model_dir, "validation_posteriors.npy"), val_probs)
    np.save(os.path.join(model_dir, "test_posteriors.npy"), test_probs)
    np.save(os.path.join(model_dir, "validation_targets.npy"), y_val)
    np.save(os.path.join(model_dir, "test_targets.npy"), y_test)
    print("Saved validation and test posteriors.")

if __name__ == "__main__":
    main()
