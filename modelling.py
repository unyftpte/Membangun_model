# modelling.py
import os, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn

DATA_DIR = "namadataset_preprocessing"
ITEMS_CSV = os.path.join(DATA_DIR, "items_clean.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "interactions_train.csv")
VALID_CSV = os.path.join(DATA_DIR, "interactions_valid.csv")

K = 10  # Top-K untuk evaluasi

def build_user_profiles(train_df, item_vecs, item_ids):
    """User profile = rata-rata vektor item yang diklik user (berbobot rating)."""
    id2row = {iid:i for i,iid in enumerate(item_ids)}
    profiles = {}
    for u, grp in train_df.groupby("user_id"):
        rows = [id2row[i] for i in grp["article_id"].values if i in id2row]
        if not rows:
            continue
        weights = grp.loc[grp["article_id"].isin(item_ids), "rating_scaled"].values[:len(rows)]
        V = item_vecs[rows]
        w = weights.reshape(-1,1) if len(weights)==V.shape[0] else np.ones((len(rows),1))
        prof = (V.multiply(w)).mean(axis=0)
        profiles[u] = prof
    return profiles

def rank_by_profile(user_id, profiles, item_vecs, item_ids, seen_items, topk=10):
    if user_id not in profiles:
        return []
    prof = profiles[user_id]
    sims = cosine_similarity(prof, item_vecs).ravel()
    order = np.argsort(-sims)
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
    # ground truth per user = item yang ada di valid (positif)
    gt = valid_df.groupby("user_id")["article_id"].apply(set).to_dict()
    users = list(gt.keys())
    precisions, recalls, aps = [], [], []
    for u in users:
        recos = recos_dict.get(u, [])[:k]
        true = gt[u]
        hits = [1 if r in true else 0 for r in recos]
        prec = sum(hits)/max(len(recos),1)
        rec  = sum(hits)/max(len(true),1)
        # AP@K
        if hits:
            cum = 0.0; correct = 0
            for i,h in enumerate(hits, start=1):
                if h==1:
                    correct += 1
                    cum += correct/i
            ap = cum/max(min(len(true),k),1)
        else:
            ap = 0.0
        precisions.append(prec); recalls.append(rec); aps.append(ap)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(aps))

def main():
    items = pd.read_csv(ITEMS_CSV)
    train = pd.read_csv(TRAIN_CSV)
    valid = pd.read_csv(VALID_CSV)

    # Vectorizer item text
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2)
    X_items = tfidf.fit_transform(items["text"])
    item_ids = items["article_id"].tolist()

    # User profiles dari train
    profiles = build_user_profiles(train, X_items, item_ids)

    # Buat dataset supervised sederhana untuk klasifikasi klik (pos-neg sampling)
    # Positif = baris train; Negatif = sample item yg tidak diklik user
    rng = np.random.default_rng(42)
    idset = set(item_ids)
    rows = []
    for u, grp in train.groupby("user_id"):
        clicked = set(grp["article_id"])
        # positif
        for iid, r in zip(grp["article_id"], grp["rating_scaled"]):
            rows.append((u, iid, 1))
        # negatif sample
        neg_cands = list(idset - clicked)
        neg_take = min(len(clicked), 20)  # batasi
        for iid in rng.choice(neg_cands, size=neg_take, replace=False):
            rows.append((u, int(iid), 0))
    sup = pd.DataFrame(rows, columns=["user_id","article_id","label"])

    # Fitur = cosine(user_profile, item_vector)
    id2row = {iid:i for i,iid in enumerate(item_ids)}
    feats = []
    labels = []
    for u,iid,l in sup.itertuples(index=False):
        if u not in profiles or iid not in id2row:
            continue
        sim = cosine_similarity(profiles[u], X_items[id2row[iid]]).ravel()[0]
        feats.append([sim]); labels.append(l)
    X = np.array(feats); y = np.array(labels)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # local
    mlflow.set_experiment("article-reco-basic")

    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="CBF-LogReg-autolog"):
        clf = LogisticRegression(max_iter=1000, n_jobs=None)
        clf.fit(Xtr, ytr)
        _ = clf.score(Xte, yte)  # metric akan terekam oleh autolog

        # Evaluasi ranking di valid melalui profile similarity (tanpa classifier),
        # sebab pipeline rekomendasinya pakai similarity.
        # (Tetap oke untuk submission Basic; tuning+manual di file satunya)
        seen_by_user = train.groupby("user_id")["article_id"].apply(set).to_dict()
        recos = {}
        for u in valid["user_id"].unique():
            recos[u] = rank_by_profile(u, profiles, X_items, item_ids, seen_by_user.get(u,set()), topk=K)
        p, r, m = precision_recall_map_at_k(valid, recos, k=K)
        # log manual tambahan sekadar info
        mlflow.log_metric(f"precision@{K}", p)
        mlflow.log_metric(f"recall@{K}", r)
        mlflow.log_metric(f"map@{K}", m)

        # Simpan vectorizer & profiles sebagai artifact
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/tfidf.pkl","wb") as f: pickle.dump(tfidf, f)
        with open("artifacts/user_profiles.pkl","wb") as f: pickle.dump(profiles, f)
        mlflow.log_artifacts("artifacts", artifact_path="cbf_assets")

if __name__ == "__main__":
    main()
