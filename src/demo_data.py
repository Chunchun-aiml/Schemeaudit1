"""Synthetic work-level data with the same columns as real MPLADS exports (used when data/real/ is empty)."""
import numpy as np
import pandas as pd


def make_demo(n_mps=120, seed=7):
    rng = np.random.default_rng(seed)
    states = ["Punjab", "Haryana", "Kerala", "Odisha", "Bihar", "Gujarat", "Assam", "Karnataka"]
    rows = []
    for i in range(n_mps):
        st, name = states[i % len(states)], f"Demo MP {i+1:03d}"
        kind = rng.choice(["ok"] * 40 + ["one_agency", "round", "rushed", "stalled"]) if i > 5 else ["one_agency", "round", "rushed", "stalled", "ok", "ok"][i]
        for j in range(int(rng.integers(15, 55))):
            amt = float(rng.choice([2, 3, 5, 7.5, 10, 12.5]) * 100000)
            if kind != "round" or rng.random() > .9:
                amt += int(rng.integers(100, 99000))
            ag = "District PWD" if kind == "one_agency" and rng.random() < .93 else f"Agency {rng.integers(1, 18)}"
            prop = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
            lag = int(rng.integers(0, 3)) if kind == "rushed" else int(rng.integers(10, 80))
            done = rng.random() < (0.08 if kind == "stalled" else 0.7)
            rows.append(dict(state=st, mp_name=name, sanction_amount=amt, implementing_agency_name=ag,
                             work_name=f"Work {j+1} in {st}", unique_work_number=f"WS/{i}/{j}",
                             work_status="Completed" if done else "Financial Sanction Approved",
                             date_of_administrative_approval=(prop + pd.Timedelta(days=lag)).strftime("%d-%m-%Y"),
                             date_of_receipt_of_work_proposal_from_mp=prop.strftime("%d-%m-%Y")))
    return pd.DataFrame(rows)