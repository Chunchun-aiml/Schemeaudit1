import glob, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from config import DATA

REQUIRED = ["state", "mp_name", "sanction_amount", "implementing_agency_name",
            "work_status", "unique_work_number",
            "date_of_administrative_approval",
            "date_of_receipt_of_work_proposal_from_mp"]
FEATS = ["n_works", "avg_sanction", "sanction_cv", "top_agency_share",
         "agencies_per_work", "median_days_to_approve", "completion_rate",
         "round_amount_share", "avg_sanction_vs_state"]
LOGCOLS = ["n_works", "avg_sanction"]


def normalise(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_works():
    files = glob.glob(str(DATA / "real" / "*.csv"))
    if not files:
        raise ValueError("No CSV files in data/real/")
    return normalise(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))


def clean(df):
    df = df.drop_duplicates("unique_work_number").copy()
    df["sanction_amount"] = pd.to_numeric(df["sanction_amount"], errors="coerce")
    df = df[df["sanction_amount"] > 0].copy()
    appr = pd.to_datetime(df["date_of_administrative_approval"], dayfirst=True, errors="coerce")
    prop = pd.to_datetime(df["date_of_receipt_of_work_proposal_from_mp"], dayfirst=True, errors="coerce")
    d = (appr - prop).dt.days
    df["days_to_approve"] = d.where(d >= 0)          # negative = bad data, ignore
    df["is_done"] = df["work_status"].astype(str).str.lower().str.contains("complet|closed")
    df["is_round"] = (df["sanction_amount"] % 100000 == 0)
    return df


def mp_features(df, min_works=10):
    g = df.groupby(["mp_name", "state"])
    top_share = g["implementing_agency_name"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    m = pd.DataFrame({
        "n_works": g.size(),
        "avg_sanction": g["sanction_amount"].mean(),
        "sanction_cv": g["sanction_amount"].std() / g["sanction_amount"].mean(),
        "top_agency_share": top_share,
        "agencies_per_work": g["implementing_agency_name"].nunique() / g.size(),
        "median_days_to_approve": g["days_to_approve"].median(),
        "completion_rate": g["is_done"].mean(),
        "round_amount_share": g["is_round"].mean(),
    }).reset_index()
    m = m[m["n_works"] >= min_works].copy()
    m["avg_sanction_vs_state"] = m["avg_sanction"] / m.groupby("state")["avg_sanction"].transform("median")
    m[FEATS] = m[FEATS].fillna(m[FEATS].median())
    return m.reset_index(drop=True)


def score(m, contamination=0.05):
    X = m[FEATS].copy()
    X[LOGCOLS] = np.log1p(X[LOGCOLS])
    Z = pd.DataFrame(RobustScaler().fit_transform(X), columns=FEATS)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42).fit(Z)
    m["anomaly_score"] = -iso.score_samples(Z)
    m["flagged"] = (iso.predict(Z) == -1).astype(int)
    why = lambda r: "; ".join(f"{c} {'high' if r[c] > 0 else 'low'}"
                              for c in r.abs().sort_values(ascending=False).index[:3])
    m["reasons"] = Z.apply(why, axis=1)
    return m.sort_values("anomaly_score", ascending=False)


if __name__ == "__main__":
    try:
        works = clean(load_works())
    except ValueError as e:
        sys.exit(str(e))
    out = score(mp_features(works))
    out.to_csv(DATA / "real_scored.csv", index=False)
    print(out[["mp_name", "state", "n_works", "anomaly_score", "reasons"]].head(20).to_string())