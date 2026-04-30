import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# ── 1. Load real Telco dataset ─────────────────────────────────────────────
def load_data():
    path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop("customerID", axis=1)
    df.columns = df.columns.str.lower()
    return df


# ── 2. Feature engineering ─────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["avg_monthly_spend"]   = df["totalcharges"] / (df["tenure"] + 1)
    df["high_value_customer"] = (df["monthlycharges"] > df["monthlycharges"].median()).astype(int)
    df["long_tenure"]         = (df["tenure"] > 24).astype(int)
    df["new_customer"]        = (df["tenure"] < 6).astype(int)
    df["service_count"]       = (
        (df["phoneservice"]    == "Yes").astype(int) +
        (df["onlinesecurity"]  == "Yes").astype(int) +
        (df["onlinebackup"]    == "Yes").astype(int) +
        (df["techsupport"]     == "Yes").astype(int) +
        (df["streamingtv"]     == "Yes").astype(int) +
        (df["streamingmovies"] == "Yes").astype(int)
    )
    df["has_internet"]       = (df["internetservice"] != "No").astype(int)
    df["auto_payment"]       = df["paymentmethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    df["charge_per_service"] = df["monthlycharges"] / (df["service_count"] + 1)
    df["tenure_x_charges"]   = df["tenure"] * df["monthlycharges"]
    return df


# ── 3. Preprocessor ────────────────────────────────────────────────────────
def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# ── 4. Train & evaluate ────────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=10,
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=300,
                                                           learning_rate=0.05,
                                                           max_depth=4, random_state=42),
    }
    results, pipelines = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy":         round(accuracy_score(y_test, y_pred),  4),
            "roc_auc":          round(roc_auc_score(y_test, y_proba),  4),
            "precision":        round(precision_score(y_test, y_pred), 4),
            "recall":           round(recall_score(y_test, y_pred),    4),
            "f1":               round(f1_score(y_test, y_pred),        4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        pipelines[name] = pipe
        print(f"  {name:25s} | AUC: {results[name]['roc_auc']:.4f} | "
              f"F1: {results[name]['f1']:.4f} | Acc: {results[name]['accuracy']:.4f}")
    return results, pipelines


# ── 5. Permutation-based feature importance ────────────────────────────────
def get_feature_importance(best_pipeline, X_test, y_test, num_cols, cat_cols):
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model        = best_pipeline.named_steps["model"]
    try:
        cat_feat_names = (
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(cat_cols).tolist()
        )
    except Exception:
        cat_feat_names = cat_cols
    feature_names = num_cols + cat_feat_names

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    X_test_tr = preprocessor.transform(X_test)
    perm = permutation_importance(model, X_test_tr, y_test,
                                   n_repeats=10, random_state=42, n_jobs=-1)

    n = min(len(feature_names), len(importances), len(perm.importances_mean))
    fi_df = pd.DataFrame({
        "feature":          feature_names[:n],
        "model_importance": importances[:n],
        "perm_importance":  perm.importances_mean[:n],
    })
    max_perm = fi_df["perm_importance"].max()
    fi_df["perm_importance_norm"] = (
        (fi_df["perm_importance"] / max_perm).round(4) if max_perm > 0
        else fi_df["model_importance"]
    )
    fi_df = fi_df.sort_values("perm_importance", ascending=False).head(20)
    return fi_df.to_dict(orient="records")


# ── 6. Business stats ─────────────────────────────────────────────────────
def compute_business_stats(df):
    stats = {}
    stats["churn_by_contract"] = (
        df.groupby("contract")["churn"].agg(["mean","count"])
        .rename(columns={"mean":"churn_rate","count":"customers"})
        .reset_index().to_dict(orient="records")
    )
    stats["churn_by_internet"] = (
        df.groupby("internetservice")["churn"].agg(["mean","count"])
        .rename(columns={"mean":"churn_rate","count":"customers"})
        .reset_index().to_dict(orient="records")
    )
    df2 = df.copy()
    df2["tenure_bucket"] = pd.cut(df2["tenure"],
        bins=[0,6,12,24,48,72], labels=["0-6m","6-12m","1-2yr","2-4yr","4-6yr"])
    stats["churn_by_tenure"] = (
        df2.groupby("tenure_bucket", observed=True)["churn"].mean()
        .reset_index().rename(columns={"churn":"churn_rate"}).to_dict(orient="records")
    )
    stats["churn_by_payment"] = (
        df.groupby("paymentmethod")["churn"].mean()
        .reset_index().rename(columns={"churn":"churn_rate"}).to_dict(orient="records")
    )
    df["svc"] = df["service_count"] if "service_count" in df.columns else 0
    stats["churn_by_service_count"] = (
        df.groupby("svc")["churn"].mean()
        .reset_index().rename(columns={"churn":"churn_rate","svc":"service_count"})
        .to_dict(orient="records")
    )
    churned  = df[df["churn"] == 1]
    retained = df[df["churn"] == 0]
    stats["monthly_revenue_at_risk"]  = round(churned["monthlycharges"].sum(), 2)
    stats["avg_charges_churned"]      = round(churned["monthlycharges"].mean(), 2)
    stats["avg_charges_retained"]     = round(retained["monthlycharges"].mean(), 2)
    stats["avg_tenure_churned"]       = round(churned["tenure"].mean(), 2)
    stats["avg_tenure_retained"]      = round(retained["tenure"].mean(), 2)
    stats["churn_senior"]     = round(df[df["seniorcitizen"]==1]["churn"].mean(), 4)
    stats["churn_non_senior"] = round(df[df["seniorcitizen"]==0]["churn"].mean(), 4)
    return stats


# ── 7. Main ───────────────────────────────────────────────────────────────
def main():
    os.makedirs("models", exist_ok=True)
    print("Loading real Telco dataset...")
    df = load_data()
    print(f"  Rows: {len(df):,} | Churn rate: {df['churn'].mean():.1%}")

    print("Engineering features...")
    df_feat = engineer_features(df)

    num_cols = [c for c in df_feat.select_dtypes(include=["int64","float64"]).columns
                if c != "churn"]
    cat_cols = df_feat.select_dtypes(include="object").columns.tolist()

    X = df_feat.drop(columns=["churn"])
    y = df_feat["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    print("\nTraining models:")
    results, pipelines = train_all_models(X_train, X_test, y_train, y_test, preprocessor)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model: {best_name} (AUC: {results[best_name]['roc_auc']})")

    print("Computing permutation feature importances...")
    fi = get_feature_importance(pipelines[best_name], X_test, y_test, num_cols, cat_cols)

    print("Computing business stats...")
    biz = compute_business_stats(df)

    output = {
        "results":            results,
        "best_model":         best_name,
        "feature_importance": fi,
        "num_cols":           num_cols,
        "cat_cols":           cat_cols,
        "dataset_stats": {
            "total_rows": len(df),
            "churn_rate": round(df["churn"].mean(), 4),
            "train_size": len(X_train),
            "test_size":  len(X_test),
        },
        "business_stats": biz,
    }

    with open("models/results.json", "w") as f:
        json.dump(output, f, indent=2)
    joblib.dump(pipelines[best_name], "models/best_model.pkl")
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, "models/feature_config.pkl")
    print("Saved: models/best_model.pkl + models/results.json")
    return output


if __name__ == "__main__":
    main()
