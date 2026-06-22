import joblib
from sklearn.ensemble import RandomForestClassifier

def train(X_train, y_train, **kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', None)
    random_state = kwargs.get('random_state', 42)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
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
