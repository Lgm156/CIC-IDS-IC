import joblib
import xgboost as xgb

def train(X_train, y_train, **kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 6)
    learning_rate = kwargs.get('learning_rate', 0.1)
    random_state = kwargs.get('random_state', 42)
    
    # multi:softprob outputs matrix of probabilities of shape (nsamples, nclasses)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    return model.predict(X)

def predict_proba(model, X):
    return model.predict_proba(X)

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
