import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_models(X_train, X_test, y_train, y_test):
    categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X_train.columns if c not in categorical]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = np.nan
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc
        })
        fitted[name] = (pipe, pred)
    return pd.DataFrame(results).sort_values("F1", ascending=False), fitted

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
