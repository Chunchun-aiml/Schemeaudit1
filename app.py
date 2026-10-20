import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from demo_data import make_demo
from real_pipeline import REQUIRED, clean, load_works, mp_features, normalise, score

st.set_page_config(page_title="SchemeAudit", page_icon="🏛️", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning works…")
def load_source(upload_bytes, upload_name):
    if upload_bytes is not None:
        raw, label = pd.read_csv(io.BytesIO(upload_bytes)), f"Uploaded file: {upload_name}"
    else:
        try:
            raw, label = load_works(), "Real data (data/real/)"
        except ValueError:
            raw, label = make_demo(), "DEMO data (synthetic). Upload a CSV or put real files in data/real/"
    return clean(normalise(raw)), label


@st.cache_data(show_spinner="Scoring MPs…")
def rank(works, min_works, contamination):
    m = mp_features(works, min_works=min_works)
    if m.empty:
        return m
    m = score(m, contamination=contamination).reset_index(drop=True)
    m.insert(0, "rank", m.index + 1)
    m["mp_key"] = m["mp_name"] + "|" + m["state"]
    return m


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Data & settings")
    up = st.file_uploader("Upload work-level CSV", type="csv")
    min_works = st.slider("Minimum works per MP", 5, 40, value=10)
    contamination = st.slider("Share of MPs to flag", 0.01, 0.15, value=0.05, step=0.01)
    with st.expander("Required CSV columns"):
        st.code("\n".join(REQUIRED))

st.title("🏛️ SchemeAudit: MPLADS review queue")

try:
    works, source = load_source(up.getvalue() if up else None, up.name if up else None)
except ValueError as e:
    st.error(str(e))
    st.stop()

mps = rank(works, min_works, contamination)
if mps.empty:
    st.warning("No MP has enough works. Lower 'Minimum works per MP' in the sidebar.")
    st.stop()

st.caption(f"Source: {source}")
c1, c2, c3 = st.columns(3)
c1.metric("MPs analysed", f"{len(mps):,}")
c2.metric("Works", f"{len(works):,}")
c3.metric("Flagged for review", int(mps["flagged"].sum()))

# ---------------- filters ----------------
f1, f2, f3 = st.columns([3, 2, 1])
q = f1.text_input("Search MP name", placeholder="Type part of a name…")
states = f2.multiselect("State", sorted(mps["state"].unique()))
only_flagged = f3.checkbox("Flagged only")

view = mps
if q:
    view = view[view["mp_name"].str.lower().str.contains(q.strip().lower(), regex=False)]
if states:
    view = view[view["state"].isin(states)]
if only_flagged:
    view = view[view["flagged"] == 1]

tab1, tab2, tab3 = st.tabs(["Review queue", "MP detail", "Insights"])

# ---------------- tab 1: queue ----------------
with tab1:
    st.write(f"**{len(view)}** MPs match")
    show = view[["rank", "mp_name", "state", "n_works", "avg_sanction", "top_agency_share",
                 "completion_rate", "median_days_to_approve", "anomaly_score", "flagged", "reasons"]].copy()
    show["top_agency_share"] *= 100
    show["completion_rate"] *= 100
    show["flagged"] = show["flagged"].map({1: "🚩", 0: ""})
    st.dataframe(
        show, hide_index=True,
        column_config={
            "anomaly_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0.0, max_value=float(mps["anomaly_score"].max()), format="%.3f"),
            "top_agency_share": st.column_config.NumberColumn("Top agency share", format="%.0f%%"),
            "completion_rate": st.column_config.NumberColumn("Completion", format="%.0f%%"),
            "avg_sanction": st.column_config.NumberColumn("Avg sanction", format="%.0f"),
            "median_days_to_approve": st.column_config.NumberColumn("Days to approve", format="%.0f"),
        })
    st.download_button("Download this list (CSV)", view.drop(columns=["mp_key"]).to_csv(index=False),
                       "schemeaudit_review_queue.csv", "text/csv")

# ---------------- tab 2: detail ----------------
with tab2:
    if view.empty:
        st.info("No MPs match the current filters.")
    else:
        labels = dict(zip(view["mp_key"], view["mp_name"] + " (" + view["state"] + ")"))
        key = st.selectbox("Pick an MP (type to search)", list(labels), format_func=labels.get)
        r = mps[mps["mp_key"] == key].iloc[0]
        med = mps[["n_works", "avg_sanction", "top_agency_share", "completion_rate",
                   "median_days_to_approve", "round_amount_share", "sanction_cv",
                   "avg_sanction_vs_state"]].median()
        st.subheader(f"{r['mp_name']} · {r['state']}")
        st.write(f"Rank **{int(r['rank'])}** of {len(mps)} · risk score **{r['anomaly_score']:.3f}**"
                 + (" · 🚩 flagged" if r["flagged"] else ""))
        st.warning(f"Why it ranks here: {r['reasons']}. This is a lead for human review, not proof of wrongdoing.")

        spec = [("n_works", "Works", False), ("avg_sanction", "Avg sanction", False),
                ("top_agency_share", "Top agency share", True), ("completion_rate", "Completion rate", True),
                ("median_days_to_approve", "Median days to approve", False),
                ("round_amount_share", "Round-amount share", True), ("sanction_cv", "Amount variability", False),
                ("avg_sanction_vs_state", "Cost vs state median", False)]
        cols = st.columns(4)
        for i, (k, label, is_pct) in enumerate(spec):
            v, m0 = float(r[k]), float(med[k])
            val = f"{v*100:.0f}%" if is_pct else f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"
            delta = f"{(v-m0)*100:+.0f} pts vs median" if is_pct else f"{v-m0:+,.1f} vs median"
            cols[i % 4].metric(label, val, delta, delta_color="off")

        w = works[(works["mp_name"] == r["mp_name"]) & (works["state"] == r["state"])]
        st.markdown("**Works by implementing agency**")
        st.bar_chart(w["implementing_agency_name"].value_counts().head(8))
        wcols = [c for c in ["unique_work_number", "work_name", "implementing_agency_name",
                             "sanction_amount", "work_status", "days_to_approve"] if c in w.columns]
        st.markdown("**Largest works**")
        st.dataframe(w.sort_values("sanction_amount", ascending=False)[wcols].head(200), hide_index=True)

# ---------------- tab 3: insights ----------------
with tab3:
    fl = mps[mps["flagged"] == 1]
    st.markdown("**Flagged MPs by state**")
    if fl.empty:
        st.info("Nothing flagged at the current settings.")
    else:
        st.bar_chart(fl.groupby("state").size())
    st.markdown("**Risk score distribution**")
    counts, edges = np.histogram(mps["anomaly_score"], bins=20)
    st.bar_chart(pd.Series(counts, index=np.round(edges[:-1], 2)))
    with st.expander("How scoring works"):
        st.write("Works are grouped per MP into nine features (agency concentration, approval speed, "
                 "round-number amounts, completion rate, cost vs state median and others). An Isolation Forest "
                 "scores how unusual each MP's profile is relative to peers. The reasons column lists the three "
                 "features furthest from the peer median. Compare MPs from one Lok Sabha term only, and check "
                 "that sanction_amount units are consistent before trusting rankings.")