import os
import pandas as pd
import numpy as np
import joblib
from models import xgboost_model

def main():
    print("Training XGBoost...")
    
    # Paths
    os.makedirs("models/xgboost", exist_ok=True)
    
    # Load data
    df_train = pd.read_parquet("processed/train_lower_stress.parquet")
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
    model = xgboost_model.train(X_train, y_train, random_state=42)
    
    # Save Model
    xgboost_model.save_model(model, "models/xgboost/model.joblib")
    print("Saved model to models/xgboost/model.joblib")
    
    # Predict probabilities
    val_probs = xgboost_model.predict_proba(model, X_val)
    test_probs = xgboost_model.predict_proba(model, X_test)
    
    # Save probabilities & targets
    np.save("models/xgboost/validation_posteriors.npy", val_probs)
    np.save("models/xgboost/test_posteriors.npy", test_probs)
    np.save("models/xgboost/validation_targets.npy", y_val)
    np.save("models/xgboost/test_targets.npy", y_test)
    print("Saved validation and test posteriors.")

if __name__ == "__main__":
    main()
