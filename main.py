# main.py
"""
Full pipeline replicating: "Diabetes Prediction Using Feature Selection Algorithms and
Boosting-Based Machine Learning Classifiers" (Diagnostics 2025).
Implements:
 - preprocessing: mean imputation, IQR outlier removal
 - balancing: RandomOverSampler
 - feature selection: RFE, Boruta (BorutaPy), GA (binary), PSO (binary), GWO (binary)
 - models: RandomForest, XGBoost, LightGBM with Hyperparameter Tuning
 - interpretability: SHAP
 - evaluation: accuracy, precision, recall, f1, roc_auc, confusion matrix
 - save best model as outputs/best_model.pkl
"""

import os
import time
import joblib
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import RandomOverSampler

# Boosting libs
import xgboost as xgb
import lightgbm as lgb

# Boruta
from boruta import BorutaPy

# SHAP
import shap

# -------------------------
# Utility helpers
# -------------------------
def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

def load_data(path="dataset/diabetes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# -------------------------
# Preprocessing as paper
# -------------------------
def preprocess(df: pd.DataFrame,
               target_col: str = "Outcome",
               impute_strategy: str = "mean",
               iqr_filter: bool = True,
               oversample: bool = True,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    imputer = SimpleImputer(strategy=impute_strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    if iqr_filter:
        keep_mask = np.ones(len(df), dtype=bool)
        for col in numeric_cols:
            q1 = np.percentile(df[col], 25)
            q3 = np.percentile(df[col], 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            col_mask = (df[col] >= lower) & (df[col] <= upper)
            keep_mask = keep_mask & col_mask
        df = df.loc[keep_mask].reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    if oversample:
        ros = RandomOverSampler(random_state=random_state)
        X_res, y_res = ros.fit_resample(X, y)
        X = pd.DataFrame(X_res, columns=X.columns)
        y = pd.Series(y_res, name=target_col)

    return X, y

# -------------------------
# Simple binary metaheuristics for feature selection
# -------------------------
from sklearn.model_selection import cross_val_score

def eval_mask_score(mask: np.ndarray, X: pd.DataFrame, y: pd.Series, cv=3, random_state=42) -> float:
    if mask.sum() == 0:
        return 0.0
    X_sel = X.iloc[:, mask.astype(bool)]
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    scores = cross_val_score(clf, X_sel, y, cv=cv, scoring="accuracy", n_jobs=1)
    return scores.mean()

def binary_genetic_algorithm(X: pd.DataFrame, y: pd.Series,
                             pop_size=20, generations=15, crossover_rate=0.8, mutation_rate=0.02) -> np.ndarray:
    n_features = X.shape[1]
    pop = np.random.randint(0, 2, size=(pop_size, n_features)).astype(int)
    best_mask = None
    best_score = -1.0

    for gen in range(generations):
        scores = np.array([eval_mask_score(ind, X, y) for ind in pop])
        elite_idx = np.argsort(scores)[-2:]
        new_pop = pop[elite_idx].copy()

        probs = (scores - scores.min()) + 1e-6
        probs = probs / probs.sum()
        cum = np.cumsum(probs)
        while len(new_pop) < pop_size:
            r1, r2 = np.random.rand(), np.random.rand()
            p1 = pop[np.searchsorted(cum, r1)]
            p2 = pop[np.searchsorted(cum, r2)]
            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, n_features-1)
                child = np.concatenate([p1[:point], p2[point:]])
            else:
                child = p1.copy()
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(0, n_features)
                child[idx] = 1 - child[idx]
            new_pop = np.vstack([new_pop, child])
        pop = new_pop[:pop_size]
        gen_best_idx = np.argmax(scores)
        gen_best_score = scores[gen_best_idx]
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_mask = pop[gen_best_idx].copy()
    return best_mask.astype(bool)

def binary_pso(X: pd.DataFrame, y: pd.Series, swarm_size=20, iterations=15, w=0.72, c1=1.5, c2=1.5):
    n_features = X.shape[1]
    pos = np.random.rand(swarm_size, n_features)
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_score = np.array([eval_mask_score((pos[i] > 0.5).astype(int), X, y) for i in range(swarm_size)])
    gbest_idx = np.argmax(pbest_score)
    gbest = pbest[gbest_idx].copy()

    for it in range(iterations):
        r1 = np.random.rand(swarm_size, n_features)
        r2 = np.random.rand(swarm_size, n_features)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0, 1)
        for i in range(swarm_size):
            mask = (pos[i] > 0.5).astype(int)
            score = eval_mask_score(mask, X, y)
            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest[i] = pos[i].copy()
        gbest_idx = np.argmax(pbest_score)
        gbest = pbest[gbest_idx].copy()
    return (gbest > 0.5).astype(bool)

def binary_gwo(X: pd.DataFrame, y: pd.Series, wolves=20, iterations=15):
    n_features = X.shape[1]
    pos = np.random.rand(wolves, n_features)
    scores = np.array([eval_mask_score((pos[i] > 0.5).astype(int), X, y) for i in range(wolves)])
    idx = np.argsort(scores)[::-1]
    alpha, beta, delta = pos[idx[0]].copy(), pos[idx[1]].copy(), pos[idx[2]].copy()

    for t in range(iterations):
        a = 2 * (1 - t / iterations)
        for i in range(wolves):
            r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            X1 = alpha - A1 * abs(C1 * alpha - pos[i])

            r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            X2 = beta - A2 * abs(C2 * beta - pos[i])

            r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            X3 = delta - A3 * abs(C3 * delta - pos[i])

            pos[i] = np.clip((X1 + X2 + X3) / 3.0, 0, 1)
        
        scores = np.array([eval_mask_score((p > 0.5).astype(int), X, y) for p in pos])
        idx = np.argsort(scores)[::-1]
        alpha, beta, delta = pos[idx[0]].copy(), pos[idx[1]].copy(), pos[idx[2]].copy()
    return (alpha > 0.5).astype(bool)

# -------------------------
# Feature selection wrappers
# -------------------------
def run_rfe(X: pd.DataFrame, y: pd.Series, n_features=5, random_state=42) -> List[int]:
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    return np.where(selector.support_)[0].tolist()

def run_boruta(X: pd.DataFrame, y: pd.Series, random_state=42) -> List[int]:
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators='auto', random_state=random_state)
    boruta.fit(X.values, y.values)
    return np.where(boruta.support_)[0].tolist()

# -------------------------
# Train & evaluate with Hyperparameter Tuning
# -------------------------
def train_and_evaluate_models_with_tuning(X_train, X_test, y_train, y_test, feature_names, verbose=True):
    results = {}
    
    # Define models and their parameter grids for GridSearchCV
    models_and_grids = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        ),
        "XGBoost": (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=42),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50]
            }
        )
    }

    for name, (model, grid) in models_and_grids.items():
        if verbose:
            print(f"--- Tuning {name} ---")
        
        t0 = time.time()
        
        # GridSearchCV for hyperparameter tuning
        cv = StratifiedKFold(n_splits=3)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate the best model found
        preds = best_model.predict(X_test)
        preds_proba = best_model.predict_proba(X_test)[:, 1]
        t1 = time.time()
        
        res = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, preds_proba),
            "time": t1 - t0,
            "model": best_model
        }
        results[name] = res
        if verbose:
            print(f"Best {name}: acc={res['accuracy']:.4f} f1={res['f1']:.4f} auc={res['roc_auc']:.4f} time={res['time']:.3f}s")
            print(f"Best params: {grid_search.best_params_}")

    return results

# -------------------------
# SHAP explainability
# -------------------------
def run_shap(model, X_sample: pd.DataFrame, feature_names: List[str], outpath="outputs/plots/shap_summary.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X_sample, show=False, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------------
# MAIN: orchestrate everything
# -------------------------
def main():
    ensure_dirs()
    print("Loading dataset...")
    df = load_data("dataset/diabetes.csv")
    print("Initial dataset shape:", df.shape)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Diabetes Features")
    plt.tight_layout()
    plt.savefig("outputs/plots/correlation_heatmap.png")
    plt.close()
    print("Saved correlation heatmap.")

    print("Preprocessing...")
    X, y = preprocess(df, target_col="Outcome")
    print("After preprocess shape:", X.shape)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, "outputs/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    print("Running feature selections...")
    rfe_idx = run_rfe(X_train, y_train, n_features=5)
    print("RFE selected:", [X.columns[i] for i in rfe_idx])
    
    try:
        boruta_idx = run_boruta(X_train, y_train)
        print("Boruta selected:", [X.columns[i] for i in boruta_idx])
    except Exception as e:
        print("Boruta failed:", e)
        boruta_idx = rfe_idx

    ga_mask = binary_genetic_algorithm(X_train, y_train)
    ga_idx = np.where(ga_mask)[0].tolist()
    print("GA selected:", [X.columns[i] for i in ga_idx])

    pso_mask = binary_pso(X_train, y_train)
    pso_idx = np.where(pso_mask)[0].tolist()
    print("PSO selected:", [X.columns[i] for i in pso_idx])

    gwo_mask = binary_gwo(X_train, y_train)
    gwo_idx = np.where(gwo_mask)[0].tolist()
    print("GWO selected:", [X.columns[i] for i in gwo_idx])
    
    print("Training baseline RF for SHAP...")
    baseline_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train, y_train)
    explainer = shap.TreeExplainer(baseline_rf)
    shap_values = explainer.shap_values(X_train)
    mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
    shap_rank_idx = np.argsort(mean_abs_shap)[::-1]
    print("Top SHAP features:", [X.columns[i] for i in shap_rank_idx[:6]])

    selection_sets = {
        "All": list(range(X.shape[1])),
        "RFE": rfe_idx,
        "Boruta": boruta_idx,
        "GA": ga_idx,
        "PSO": pso_idx,
        "GWO": gwo_idx,
        "TopSHAP5": shap_rank_idx[:5].tolist()
    }

    all_results = {}
    for name, idxs in selection_sets.items():
        if not idxs: continue
        print(f"\n=== Training with feature set: {name} (n={len(idxs)}) ===")
        Xtr_sel = X_train.iloc[:, idxs]
        Xte_sel = X_test.iloc[:, idxs]
        feature_names = [X.columns[i] for i in idxs]
        
        # Use the new tuning function
        res = train_and_evaluate_models_with_tuning(Xtr_sel, Xte_sel, y_train, y_test, feature_names)
        all_results[name] = res
        
        best_model_name = max(res.items(), key=lambda kv: kv[1]['f1'])[0]
        joblib.dump(res[best_model_name]['model'], f"outputs/model_{name}_{best_model_name}.pkl")
        print(f"Saved best model for {name}: model_{name}_{best_model_name}.pkl")

    best_overall = ("", 0.0, None)
    for sname, resdict in all_results.items():
        for mname, stats in resdict.items():
            if stats['f1'] > best_overall[1]:
                best_overall = (f"{sname}::{mname}", stats['f1'], stats['model'])
    
    print("\nBest overall model:", best_overall[0], "with F1:", best_overall[1])
    if best_overall[2]:
        joblib.dump(best_overall[2], "outputs/best_model.pkl")
        print("Saved overall best model to outputs/best_model.pkl")

        try:
            sname, _ = best_overall[0].split("::")
            idxs = selection_sets[sname]
            run_shap(best_overall[2], X_test.iloc[:, idxs], [X.columns[i] for i in idxs])
            print("Saved SHAP plot for best model.")
        except Exception as e:
            print("SHAP plotting error:", e)

    rows = []
    for sname, resdict in all_results.items():
        for mname, stats in resdict.items():
            rows.append({
                "feature_set": sname,
                "model": mname,
                "accuracy": stats['accuracy'],
                "precision": stats['precision'],
                "recall": stats['recall'],
                "f1": stats['f1'],
                "roc_auc": stats.get('roc_auc', 0.0),
                "train_time_s": stats['time']
            })
    repdf = pd.DataFrame(rows)
    repdf.to_csv("outputs/summary_results.csv", index=False)
    print("Saved summary report to outputs/summary_results.csv")
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
