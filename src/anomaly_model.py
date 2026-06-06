from sklearn.ensemble import IsolationForest

import config

def train_isolation_forest(X_train, contamination=config.CONTAMINATION):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train)
    return model

def predict_anomalies(model, X):
    scores = model.decision_function(X)
    labels = model.predict(X)
    return scores, labels
