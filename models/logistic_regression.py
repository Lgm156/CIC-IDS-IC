import joblib
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, **kwargs):
    # Retrieve hyperparameters with defaults
    C = kwargs.get('C', 1.0)
    max_iter = kwargs.get('max_iter', 1000)
    random_state = kwargs.get('random_state', 42)
    solver = kwargs.get('solver', 'lbfgs')
    
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver
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
