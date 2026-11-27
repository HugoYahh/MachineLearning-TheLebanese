from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import TARGET_COL

# Predict no drawdown for every observation.
def baseline_zero(n_samples):
    return [0] * n_samples

# Use each observation's VIX value as its drawdown risk score.
def baseline_vix(data):
    return list(data["vix"])

# Build B2 with median imputation, feature scaling, and class-weighted logistic regression.
def build_baseline_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),#fill in missing values with the median of each feature to avoid losing data and maintain the distribution of the features.
        ("scaler", StandardScaler()),#standardize the features to have zero mean and unit variance, which is important for logistic regression.
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)), #gives a higher weight to the minority class (drawdowns) to improve recall and reduce false negatives.
    ])

# Fit the logistic regression pipeline on the selected training features and labels.
def fit_baseline_logistic(train, feature_columns):
    x_train = train[feature_columns]
    y_train = train[TARGET_COL]

    model = build_baseline_logistic()
    model.fit(x_train, y_train)

    return model

# Return the logistic regression score for class 1 for each validation observation.
def baseline_logistic_scores(model, validation, feature_columns):
    x_val = validation[feature_columns]
    probabilities = model.predict_proba(x_val)
    return probabilities[:, 1].tolist()

