"""
Behaviour / demography classification from pre-computed acoustic features.

Uses the CNN embeddings already stored in outputs/*_vocalisation_features.parquet
(computed by the authors on the FULL Dzanga-Bai dataset), so no audio is needed.

Unit of analysis = one recording file (the contextual labels are given per file),
exactly as in 3_behaviour_demography_notebook.ipynb: features of all rumbles in a
file are averaged. Train/test splits are grouped by recording session ("opp_###")
so that clips of the same event never appear in both train and test.

Run from the repo root:
    terminal:        python 4_behaviour_classification.py
    Jupyter/Colab:   %run 4_behaviour_classification.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # no pop-up windows when run from a terminal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths (works from repo root, or from /content in Colab)
# --------------------------------------------------------------------------- #
ROOT = Path(".")
for candidate in [Path("."), Path("elephant_transfer_learning"), Path("/content/elephant_transfer_learning")]:
    if (candidate / "outputs" / "yamnet_vocalisation_features.parquet").exists():
        ROOT = candidate
        break
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
RESULTS_DIR = OUT_DIR / "behaviour_classification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MODELS = ["VGGish", "YAMNet", "BirdNET", "Perch"]
N_SPLITS = 5
SEEDS = [0, 1, 2]  # repeated grouped CV -> more stable estimates

# --------------------------------------------------------------------------- #
# 1. Labels: one row per recording file, rumble-only files (as in notebook 3)
# --------------------------------------------------------------------------- #
voc = pd.read_csv(DATA_DIR / "elephant_vocalisations.csv")
ctx = pd.read_csv(DATA_DIR / "elephant_contextual_observations.csv")

meta = voc.merge(ctx, on="filename")
meta = meta.groupby("filename").filter(lambda g: (g["call_type"] == "Rumble").all())

files = meta.drop_duplicates("filename").set_index("filename")[["Behaviour", "Distress", "Age", "Sex"]]
files["session"] = files.index.str.extract(r"^(opp_\d+)", expand=False).values
print(f"Labelled rumble files: {len(files)}  |  recording sessions: {files['session'].nunique()}")

# --------------------------------------------------------------------------- #
# 2. Features: average the embeddings of all rumbles in each file
# --------------------------------------------------------------------------- #
features = {}
for name in FEATURE_MODELS:
    emb = pd.read_parquet(OUT_DIR / f"{name.lower()}_vocalisation_features.parquet")
    emb = emb[[c for c in emb.columns if c.startswith("feature_")]]
    merged = meta[["filename", "vocalisation_id"]].merge(emb, left_on="vocalisation_id", right_index=True)
    features[name] = merged.drop(columns="vocalisation_id").groupby("filename").mean().loc[files.index]
features["All (concat)"] = pd.concat(
    [f.add_prefix(f"{n}_") for n, f in features.items()], axis=1
)

# --------------------------------------------------------------------------- #
# 3. Tasks: which labels to predict, and which (rare / unknown) classes to drop
# --------------------------------------------------------------------------- #
TASKS = {
    # Sexual (8) and Anti-Predator (2) are too rare to learn or evaluate.
    "Behaviour": dict(col="Behaviour", drop=["Unknown", "Sexual", "Anti-Predator"]),
    "Distress": dict(col="Distress", drop=["Unknown", "Not Applicable"]),
    "Age": dict(col="Age", drop=["Unknown"]),
    "Sex": dict(col="Sex", drop=["Unknown"]),
}

CLASSIFIERS = {
    "Chance (stratified)": DummyClassifier(strategy="stratified", random_state=0),
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)
    ),
    "RBF SVM": make_pipeline(
        StandardScaler(), PCA(n_components=0.95, svd_solver="full"), SVC(C=3, gamma="scale", class_weight="balanced")
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=0
    ),
}


def grouped_cv_predict(model, X, y, groups, seed):
    """Out-of-fold predictions with session-grouped, stratified folds."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        m = clone(model)
        if hasattr(m, "random_state"):
            m.set_params(random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


# --------------------------------------------------------------------------- #
# 4. Evaluate every task x feature set x classifier
# --------------------------------------------------------------------------- #
rows, best = [], {}
for task, spec in TASKS.items():
    keep = ~files[spec["col"]].isin(spec["drop"])
    y = files.loc[keep, spec["col"]].to_numpy()
    groups = files.loc[keep, "session"].to_numpy()
    counts = pd.Series(y).value_counts().to_dict()
    print(f"\n=== {task}: n={len(y)}  classes={counts}")

    for feat_name, F in features.items():
        X = F.loc[keep].to_numpy(dtype=np.float32)
        for clf_name, clf in CLASSIFIERS.items():
            if clf_name.startswith("Chance") and feat_name != "All (concat)":
                continue  # chance baseline doesn't depend on features
            f1s, bals, preds = [], [], []
            for seed in SEEDS:
                p = grouped_cv_predict(clf, X, y, groups, seed)
                f1s.append(f1_score(y, p, average="macro"))
                bals.append(balanced_accuracy_score(y, p))
                preds.append(p)
            fname = "-" if clf_name.startswith("Chance") else feat_name
            rows.append(dict(task=task, features=fname, classifier=clf_name, n=len(y),
                             macro_f1=np.mean(f1s), macro_f1_sd=np.std(f1s), balanced_acc=np.mean(bals)))
            print(f"  {fname:13s} {clf_name:20s} macro-F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
                  f"   balanced-acc {np.mean(bals):.3f}")
            if not clf_name.startswith("Chance") and np.mean(f1s) > best.get(task, {}).get("f1", -1):
                best[task] = dict(f1=np.mean(f1s), features=feat_name, clf=clf_name, y=y, pred=preds[0], keep=keep)

results = pd.DataFrame(rows).sort_values(["task", "macro_f1"], ascending=[True, False])
results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. Detailed report + confusion matrix for the best model of each task,
#    then refit it on all labelled data and save it.
# --------------------------------------------------------------------------- #
print("\n" + "=" * 78 + "\nSUMMARY (macro-F1, session-grouped 5-fold CV x 3 repeats)\n" + "=" * 78)
for task, b in best.items():
    chance = results.query("task == @task and classifier.str.startswith('Chance')", engine="python")["macro_f1"].iloc[0]
    print(f"\n### {task}: best = {b['clf']} on {b['features']} features  "
          f"(macro-F1 {b['f1']:.3f} vs chance {chance:.3f})")
    print(classification_report(b["y"], b["pred"], digits=3, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(b["y"], b["pred"], normalize="true", values_format=".2f",
                                            cmap="Blues", ax=ax, xticks_rotation=45)
    ax.set_title(f"{task} — {b['clf']} ({b['features']})\nrow-normalised, out-of-fold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{task.lower()}.png", dpi=150)
    if IN_NOTEBOOK:
        plt.show()  # inline in Jupyter/Colab; from a terminal the PNG is just saved
    plt.close(fig)

    final = clone(CLASSIFIERS[b["clf"]]).fit(features[b["features"]].loc[b["keep"]].to_numpy(np.float32), b["y"])
    joblib.dump(dict(model=final, feature_set=b["features"], classes=sorted(set(b["y"]))),
                RESULTS_DIR / f"best_{task.lower()}_model.joblib")

print(f"\nSaved results, confusion matrices and fitted models to: {RESULTS_DIR.resolve()}")
