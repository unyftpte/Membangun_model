# modelling.py (final, L2-normalized profiles + candidate pruning, fixed MLflow metric names)

import os, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import mlflow
import mlflow.sklearn

DATA_DIR = "namadataset_preprocessing"
ITEMS_CSV = os.path.join(DATA_DIR, "items_clean.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "interactions_train.csv")
VALID_CSV = os.path.join(DATA_DIR, "interactions_valid.csv")

K = 10           # Top-K untuk evaluasi
PRUNE_TOP = 1000 # dari 300 -> 1000 (lebih banyak kandidat)

def log_metric_safe(name: str, value: float):
    """Pastikan nama metrik valid untuk MLflow (tidak mengandung '@')."""
    mlflow.log_metric(name.replace("@", "_at_"), float(value))

def ensure_text_column(items: pd.DataFrame) -> pd.DataFrame:
    """Pastikan kolom 'text' ada & tidak kosong (fallback dari title/content/category)."""
    if "text" not in items.columns:
        for c in ["title", "content", "category"]:
            if c not in items.columns:
                items[c] = ""
        items["text"] = (
            items["title"].fillna("") + " " +
            items["content"].fillna("") + " " +
            items["category"].fillna("")
        )

    # Normalisasi ringan
    items["text"] = (
        items["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Buang baris benar-benar kosong
    items = items[items["text"].str.len() > 0].copy()

    # --- categorical boost (meningkatkan sinyal tematik dari kategori) ---
    if "category" in items.columns:
        items["cat_boost"] = ((" " + items["category"].fillna("")) * 3).str.strip()
        items["text"] = (items["text"] + " " + items["cat_boost"]).str.strip()
        items.drop(columns=["cat_boost"], inplace=True)

    return items

def vectorize_items(items: pd.DataFrame):
    """TF-IDF dengan guard agar tidak empty vocabulary (lebih ketat utk kurangi noise)."""
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,            # dari 1 -> 3
        max_df=0.85,         # buang term terlalu umum
        max_features=80000,  # naikkan kapasitas
        stop_words="english",
        token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
        sublinear_tf=True,
    )
    try:
        X = tfidf.fit_transform(items["text"])
    except ValueError:
        # Fallback paling permisif
        tfidf = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=1,
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
        )
        X = tfidf.fit_transform(items["text"])

    if X.shape[1] == 0:
        # Jika masih kosong, pakai fitur dummy panjang teks
        items["_len"] = items["text"].str.len().astype(float)
        X = sp.csr_matrix(items[["_len"]].values)
        class _Dummy:
            def transform(self, a):  # placeholder
                return X
        tfidf = _Dummy()
    return tfidf, X

def build_user_profiles(train_df, item_vecs, item_ids):
    """User profile = rata-rata vektor item yang diklik user (berbobot rating) + L2 normalize."""
    id2row = {iid: i for i, iid in enumerate(item_ids)}
    profiles = {}
    for u, grp in train_df.groupby("user_id"):
        rows = [id2row[i] for i in grp["article_id"].values if i in id2row]
        if not rows:
            continue
        weights = grp.loc[grp["article_id"].isin(item_ids), "rating_scaled"].values[: len(rows)]
        V = item_vecs[rows]  # sparse (n_clicked x d)
        w = weights.reshape(-1, 1) if len(weights) == V.shape[0] else np.ones((len(rows), 1))
        prof = (V.multiply(w)).mean(axis=0)     # -> 1 x d (np.matrix-like)
        prof = np.asarray(prof).ravel().reshape(1, -1)
        prof = sp.csr_matrix(prof)
        profiles[u] = normalize(prof, norm="l2", copy=False)
    return profiles

def rank_by_profile(user_id, profiles, item_vecs, item_ids, seen_items, topk=10):
    if user_id not in profiles:
        return []
    prof = profiles[user_id]               # CSR 1 x d
    sims = cosine_similarity(prof, item_vecs).ravel()
    order = np.argsort(-sims)[:PRUNE_TOP]  # pruning kandidat
    reco = []
    for idx in order:
        iid = item_ids[idx]
        if iid in seen_items:
            continue
        reco.append(int(iid))
        if len(reco) >= topk:
            break
    return reco

def precision_recall_map_at_k(valid_df, recos_dict, k=10):
    gt = valid_df.groupby("user_id")["article_id"].apply(set).to_dict()
    users = list(gt.keys())
    precisions, recalls, aps = [], [], []
    for u in users:
        recos = recos_dict.get(u, [])[:k]
        true = gt[u]
        hits = [1 if r in true else 0 for r in recos]
        prec = sum(hits) / max(len(recos), 1)
        rec = sum(hits) / max(len(true), 1)
        # AP@K
        if hits:
            cum = 0.0
            correct = 0
            for i, h in enumerate(hits, start=1):
                if h == 1:
                    correct += 1
                    cum += correct / i
            ap = cum / max(min(len(true), k), 1)
        else:
            ap = 0.0
        precisions.append(prec)
        recalls.append(rec)
        aps.append(ap)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(aps))

def main():
    items = pd.read_csv(ITEMS_CSV)
    train = pd.read_csv(TRAIN_CSV)
    valid = pd.read_csv(VALID_CSV)

    # Pastikan 'text' siap
    items = ensure_text_column(items)

    # Vectorize item text
    tfidf, X_items = vectorize_items(items)
    item_ids = items["article_id"].tolist()
    print(f"[INFO] TF-IDF shape: {getattr(X_items, 'shape', np.asarray(X_items).shape)} | items: {len(item_ids)}")

    # User profiles dari train
    profiles = build_user_profiles(train, X_items, item_ids)

    # Siapkan dataset supervised sederhana (pos-neg)
    rng = np.random.default_rng(42)
    idset = set(item_ids)
    rows = []
    for u, grp in train.groupby("user_id"):
        clicked = set(grp["article_id"])
        # positif
        for iid, _ in zip(grp["article_id"], grp["rating_scaled"]):
            rows.append((u, iid, 1))
        # negatif sample
        neg_cands = list(idset - clicked)
        if len(neg_cands) == 0:
            continue
        neg_take = min(len(clicked), 20)
        take = min(neg_take, len(neg_cands))
        for iid in rng.choice(neg_cands, size=take, replace=False):
            rows.append((u, int(iid), 0))
    sup = pd.DataFrame(rows, columns=["user_id", "article_id", "label"])

    # Fitur = cosine(user_profile, item_vector)
    id2row = {iid: i for i, iid in enumerate(item_ids)}
    feats, labels = [], []
    for u, iid, l in sup.itertuples(index=False):
        if u not in profiles or iid not in id2row:
            continue
        sim = cosine_similarity(profiles[u], X_items[id2row[iid]]).ravel()[0]
        feats.append([sim]); labels.append(l)
    X = np.array(feats); y = np.array(labels)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # MLflow local (kalau server tak ada, tetap ke ./mlruns)
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    except Exception:
        pass
    mlflow.set_experiment("article-reco-basic")

    # opsional: autolog bisa dimatikan kalau mau lebih minimal
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="CBF-LogReg-autolog"):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        _ = clf.score(Xte, yte)  # autolog

        # Evaluasi Top-K via similarity
        seen_by_user = train.groupby("user_id")["article_id"].apply(set).to_dict()
        recos = {
            u: rank_by_profile(u, profiles, X_items, item_ids, seen_by_user.get(u, set()), topk=K)
            for u in valid["user_id"].unique()
        }
        p, r, m = precision_recall_map_at_k(valid, recos, k=K)

        # nama metrik valid utk MLflow
        log_metric_safe(f"precision@{K}", p)  # -> precision_at_10
        log_metric_safe(f"recall@{K}", r)     # -> recall_at_10
        log_metric_safe(f"map@{K}", m)        # -> map_at_10

        # artifacts
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/tfidf.pkl", "wb") as f: pickle.dump(tfidf, f)
        with open("artifacts/user_profiles.pkl", "wb") as f: pickle.dump(profiles, f)
        mlflow.log_artifacts("artifacts", artifact_path="cbf_assets")

        print(f"[OK] precision@{K}={p:.4f} | recall@{K}={r:.4f} | MAP@{K}={m:.4f}")

if __name__ == "__main__":
    main()

