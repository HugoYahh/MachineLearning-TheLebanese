from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import TARGET_COL

def baseline_zero(n_samples):
    return [0] * n_samples

def baseline_vix(data):
    return list(data["vix"])

def build_baseline_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])

def fit_baseline_logistic(train, feature_columns):
    x_train = train[feature_columns]
    y_train = train[TARGET_COL]

    model = build_baseline_logistic()
    model.fit(x_train, y_train)

    return model

def baseline_logistic_scores(model, validation, feature_columns):
    x_val = validation[feature_columns]
    probabilities = model.predict_proba(x_val)
    return probabilities[:, 1].tolist()

