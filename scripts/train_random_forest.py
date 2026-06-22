import os
import pandas as pd
import numpy as np
import joblib
from models import random_forest

def main():
    print("Training Random Forest...")
    
    # Paths
    os.makedirs("models/random_forest", exist_ok=True)
    
    # Load data
    df_train = pd.read_parquet("processed/train.parquet")
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
    model = random_forest.train(X_train, y_train, random_state=42)
    
    # Save Model
    random_forest.save_model(model, "models/random_forest/model.joblib")
    print("Saved model to models/random_forest/model.joblib")
    
    # Predict probabilities
    val_probs = random_forest.predict_proba(model, X_val)
    test_probs = random_forest.predict_proba(model, X_test)
    
    # Save probabilities & targets
    np.save("models/random_forest/validation_posteriors.npy", val_probs)
    np.save("models/random_forest/test_posteriors.npy", test_probs)
    np.save("models/random_forest/validation_targets.npy", y_val)
    np.save("models/random_forest/test_targets.npy", y_test)
    print("Saved validation and test posteriors.")

if __name__ == "__main__":
    main()
