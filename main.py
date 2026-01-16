import pandas as pd
import numpy as np

def load_data_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

df_train = load_data_jsonl("data/multinli_1.0_train.jsonl")
df_dev_matched = load_data_jsonl("data/multinli_1.0_dev_matched.jsonl")
df_dev_mismatched = load_data_jsonl("data/multinli_1.0_dev_mismatched.jsonl")

label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

def prepare_df(df):
    df = df[df["gold_label"].isin(label2id)]
    sent1 = df["sentence1"].values
    sent2 = df["sentence2"].values
    labels = df["gold_label"].map(label2id).values
    return sent1, sent2, labels

X_train_s1, X_train_s2, y_train = prepare_df(df_train)
X_dev_m_s1, X_dev_m_s2, y_dev_m = prepare_df(df_dev_matched)
X_dev_mm_s1, X_dev_mm_s2, y_dev_mm = prepare_df(df_dev_mismatched)

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2
)

# FIT on TRAIN sentence1
vectorizer.fit(X_train_s1)
X_train_s1_vec = vectorizer.transform(X_train_s1)
X_train_s2_vec = vectorizer.transform(X_train_s2)

# Concatenate the two vectors horizontally
X_train_vec = hstack([X_train_s1_vec, X_train_s2_vec])

# TRANSFORM on DEV
X_dev_m_s1_vec = vectorizer.transform(X_dev_m_s1)
X_dev_m_s2_vec = vectorizer.transform(X_dev_m_s2)
X_dev_m_vec = hstack([X_dev_m_s1_vec, X_dev_m_s2_vec])

X_dev_mm_s1_vec = vectorizer.transform(X_dev_mm_s1)
X_dev_mm_s2_vec = vectorizer.transform(X_dev_mm_s2)
X_dev_mm_vec = hstack([X_dev_mm_s1_vec, X_dev_mm_s2_vec])

# print("Train vector shape:", X_train_vec.shape)

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

svm = LinearSVC(
    C=1.0,
    max_iter=5000
)

svm.fit(X_train_vec, y_train)

def evaluate(name, X_vec, y_true):
    y_pred = svm.predict(X_vec)
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} accuracy: {acc:.4f}")

evaluate("Dev matched", X_dev_m_vec, y_dev_m)
evaluate("Dev mismatched", X_dev_mm_vec, y_dev_mm)