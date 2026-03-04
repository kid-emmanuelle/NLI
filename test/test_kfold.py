import pandas as pd
import numpy as np
from scipy.sparse import hstack

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def load_data_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

# Load data
df_train = load_data_jsonl("data/multinli_1.0_train.jsonl")[:10000]  # limit to 10k for faster experimentation
df_dev_matched = load_data_jsonl("data/multinli_1.0_dev_matched.jsonl")
df_dev_mismatched = load_data_jsonl("data/multinli_1.0_dev_mismatched.jsonl")

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

def prepare_df(df):
    df = df[df["gold_label"].isin(label2id)].copy()
    s1 = df["sentence1"].astype(str).values
    s2 = df["sentence2"].astype(str).values
    y = df["gold_label"].map(label2id).astype(int).values
    return s1, s2, y

X_train_s1, X_train_s2, y_train = prepare_df(df_train)
X_dev_m_s1, X_dev_m_s2, y_dev_m = prepare_df(df_dev_matched)
X_dev_mm_s1, X_dev_mm_s2, y_dev_mm = prepare_df(df_dev_mismatched)

def kfold_cv_choose_C(s1, s2, y, C_values, k=5, seed=42, max_features=50000):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    results = {}
    for C in C_values:
        scores = []
        for train_idx, val_idx in skf.split(s1, y):
            s1_tr, s2_tr, y_tr = s1[train_idx], s2[train_idx], y[train_idx]
            s1_va, s2_va, y_va = s1[val_idx], s2[val_idx], y[val_idx]

            # IMPORTANT: fit vectorizer ONLY on fold-train (avoid leakage)
            vec = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=max_features,
                min_df=2
            )
            vec.fit(np.concatenate([s1_tr, s2_tr]))

            X_tr = hstack([vec.transform(s1_tr), vec.transform(s2_tr)])
            X_va = hstack([vec.transform(s1_va), vec.transform(s2_va)])

            clf = LinearSVC(C=C, max_iter=5000)
            clf.fit(X_tr, y_tr)

            pred = clf.predict(X_va)
            scores.append(accuracy_score(y_va, pred))

        results[C] = (float(np.mean(scores)), float(np.std(scores)))
    return results

# 1) Cross-validation to pick C (lambda)
C_values = [0.1, 0.3, 1.0, 3.0, 10.0]
cv_results = kfold_cv_choose_C(X_train_s1, X_train_s2, y_train, C_values, k=5)

print("CV results (mean ± std):")
for C, (m, s) in sorted(cv_results.items(), key=lambda x: x[0]):
    print(f"  C={C:<4}  {m:.4f} ± {s:.4f}")

best_C = max(cv_results, key=lambda c: cv_results[c][0])
print("\nBest C:", best_C)

# 2) Train final model on FULL train with best_C
final_vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2
)
final_vec.fit(np.concatenate([X_train_s1, X_train_s2]))

X_train_vec = hstack([final_vec.transform(X_train_s1), final_vec.transform(X_train_s2)])
X_dev_m_vec  = hstack([final_vec.transform(X_dev_m_s1),  final_vec.transform(X_dev_m_s2)])
X_dev_mm_vec = hstack([final_vec.transform(X_dev_mm_s1), final_vec.transform(X_dev_mm_s2)])

final_svm = LinearSVC(C=best_C, max_iter=5000)
final_svm.fit(X_train_vec, y_train)

def evaluate(name, X_vec, y_true):
    y_pred = final_svm.predict(X_vec)
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} accuracy: {acc:.4f}")

print("\nFinal eval on dev sets:")
evaluate("Dev matched", X_dev_m_vec, y_dev_m)
evaluate("Dev mismatched", X_dev_mm_vec, y_dev_mm)
