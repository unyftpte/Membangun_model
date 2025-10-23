# serve_reco.py
import os, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "namadataset_preprocessing"
ITEMS_CSV = os.path.join(DATA_DIR, "items_clean.csv")
ASSET_DIR = "models/cbf_best"  # tempat kamu taruh tfidf.pkl & user_profiles.pkl

def load_assets():
    with open(os.path.join(ASSET_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(ASSET_DIR, "user_profiles.pkl"), "rb") as f:
        profiles = pickle.load(f)
    return tfidf, profiles

def ensure_text(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("title","content","category","text"):
        if c not in df.columns: df[c] = ""
    if (df["text"].fillna("").str.len()==0).any():
        m = df["text"].fillna("")=="" 
        df.loc[m,"text"] = (df.loc[m,"title"] + " " + df.loc[m,"category"]).str.lower()
    return df

def recommend(user_id: int, topk=10, min_similarity=0.1):
    items = ensure_text(pd.read_csv(ITEMS_CSV))
    item_ids = items["article_id"].tolist()

    tfidf, profiles = load_assets()
    X_items = tfidf.transform(items["text"])  # sparse (n_items, d)

    if user_id not in profiles:
        return []

    prof = profiles[user_id]                # CSR (1, d)
    sims = cosine_similarity(prof, X_items).ravel()
    order = np.argsort(-sims)
    reco = [int(item_ids[i]) for i in order if sims[i] >= float(min_similarity)]
    return reco[:topk]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min_sim", type=float, default=0.1)
    args = ap.parse_args()
    print(recommend(args.user_id, args.topk, args.min_sim))
