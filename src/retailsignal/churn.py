"""Compare interpretable and tree-based customer churn classifiers."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "churned"
DROP_COLUMNS = {"customer_id", TARGET}


def train_churn_models(modelled_customers_path: str | Path, reports_dir: str | Path,
                       models_dir: str | Path, random_state: int = 42) -> dict[str, Path]:
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(modelled_customers_path)
    y = data[TARGET].astype(int)
    X = data.drop(columns=[column for column in DROP_COLUMNS if column in data.columns])
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=12, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=220, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=random_state),
    }
    rows = []
    fitted = {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, prediction),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
        })
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    names = best_model.named_steps["preprocess"].get_feature_names_out()
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = abs(estimator.coef_[0])
    importance = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).head(20)
    metrics_path = reports_dir / "churn_model_metrics.csv"
    importance_path = reports_dir / "churn_feature_importance.csv"
    model_path = models_dir / "best_churn_model.joblib"
    metrics.to_csv(metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    joblib.dump(best_model, model_path)
    return {"metrics": metrics_path, "importance": importance_path, "model": model_path}

