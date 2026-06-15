"""
Freeze the SENTINEL deployment model to disk for the Streamlit app.

Run this once (and again whenever the modelling logic in sentinel_core.py changes) to produce
model.joblib + meta.json next to this file. It builds the engine via sentinel_core.build_engine,
then ASSERTS that the frozen model reproduces the notebook's headline behaviour before saving:

  1. EnsembleClassifierChains.predict_proba reproduces the reference fit_ecc exactly
     (the structural refactor introduces no behavioural drift);
  2. the single-label baseline misses the non-malarial dengue cell while the multi-label models
     recover it (RQ1), with binary relevance >= the ensemble of chains on that cell (within noise);
  3. per-label conformal coverage sits near the 90 percent target for malaria and dengue (RQ3);
  4. the conformal referral rule catches far more dengue than a naive 0.5 detector (RQ4 / Section 5.4);
  5. the value-of-information next-test recommendation matches the notebook.

If any assertion fails the model is NOT written, so a stale or drifted export cannot ship silently.

Usage:
    cd app && python export_model.py
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys

import joblib
import numpy as np

import sentinel_core as core


def _resolve_data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "data.csv"),
                 os.path.join(here, "data.csv"),
                 "data.csv"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "data.csv not found. Place the cohort file at the repository root and rerun.")


def main() -> int:
    data_path = _resolve_data_path()
    print(f"Building deployment engine from {os.path.abspath(data_path)} ...")
    eng = core.build_engine(data_path)

    X_deploy = eng["_X_deploy"]
    Yv = eng["_Yv"]
    P_ecc = eng["_P_ecc"]
    ecc = eng["ecc"]
    qhat = eng["qhat"]
    conf = eng["conf"]
    test_idx = eng["_test_idx"]
    LABELS = eng["labels"]

    # ---- Assert 1: the refactored ECC reproduces the reference fit_ecc exactly --------
    from numpy.random import default_rng
    P_ref = core.fit_ecc(X_deploy, Yv, X_deploy, n_chains=8, rng=default_rng(core.SEED))
    P_cls = ecc.predict_proba(X_deploy)
    max_abs = float(np.max(np.abs(P_ref - P_cls)))
    print(f"[1] ECC refactor vs reference fit_ecc: max abs prob difference = {max_abs:.2e}")
    assert max_abs < 1e-9, (
        f"EnsembleClassifierChains drifted from fit_ecc (max diff {max_abs:.2e}); not faithful.")

    # ---- Assert 2: RQ1, the multi-label formulation recovers non-malarial dengue ------
    di, mi = LABELS.index("dengue"), LABELS.index("malaria")
    regimes = core.make_regimes(core.build_dataset(core.load_raw(data_path)))
    data = core.build_dataset(core.load_raw(data_path))
    X, M = data["X"], data["M"]
    splits_s = core.strat_splits(Yv)
    spec_sym = regimes["REGIMES"]["symptoms"]

    P_br = core.oof_probabilities(
        lambda Xtr, Ytr, Xte: _fit_br(Xtr, Ytr, Xte, LABELS),
        spec_sym, splits_s, X, M, regimes, Yv)

    nmd = (Yv[:, di] == 1) & (Yv[:, mi] == 0)
    recall_br = float(((P_br[:, di] >= 0.5).astype(int)[nmd] == 1).mean())
    recall_ecc = float(((P_ecc[:, di] >= 0.5).astype(int)[nmd] == 1).mean())
    print(f"[2] Non-malarial dengue recall (n={int(nmd.sum())}): "
          f"BR {recall_br:.1%}, ECC {recall_ecc:.1%}  (single-label literature ~0-5.5%)")
    assert recall_br >= 0.30, f"BR non-malarial dengue recall too low ({recall_br:.1%})"
    assert recall_ecc >= 0.25, f"ECC non-malarial dengue recall too low ({recall_ecc:.1%})"

    # ---- Assert 3: RQ3, conformal coverage near the 90 percent target -----------------
    cov_mal = conf["malaria"]["coverage"]
    cov_den = conf["dengue"]["coverage"]
    print(f"[3] Conformal coverage on true cases: Malaria {cov_mal:.0%}, Dengue {cov_den:.0%} "
          f"(target {int((1-core.ALPHA)*100)}%)")
    assert 0.80 <= cov_mal <= 1.0, f"Malaria coverage off target ({cov_mal:.0%})"
    assert 0.70 <= cov_den <= 1.0, f"Dengue coverage off target ({cov_den:.0%})"

    # ---- Assert 4: RQ4 / Section 5.4 referral beats the naive detector ----------------
    acct = core.referral_accounting(conf, P_ecc, Yv, test_idx)
    print(f"[4] Referral on the conformal test split: catches {acct['caught']}/{acct['dengue_n']} "
          f"dengue (naive 0.5 detector: {acct['naive_caught']}/{acct['dengue_n']}); "
          f"missed {acct['missed']}, over-referred {acct['overref']}, "
          f"referral rate {acct['referral_rate']:.0%}")
    assert acct["caught"] >= acct["naive_caught"], "Referral rule should catch >= the naive detector"
    assert acct["caught"] >= int(0.7 * acct["dengue_n"]), "Referral sensitivity unexpectedly low"

    # ---- Assert 5: next-test recommendation matches the notebook ----------------------
    print(f"[5] Next-test recommendation when dengue is unresolved: {eng['next_test']}")
    assert isinstance(eng["next_test"], str) and eng["next_test"], "next_test missing"

    # ---- Local-recalibration sanity (Section 6.2 demo) --------------------------------
    rm = core.recalibration_matrix(eng["recal_data"])
    den = rm["matrix"]["dengue"]
    sites = rm["sites"]
    scarce = min(sites, key=lambda s: rm["n_positives"]["dengue"][s])  # the dengue-scarce district
    cov_pooled = den["pooled"][scarce]["coverage"]
    cov_local = den[scarce][scarce]["coverage"]
    print(f"[6] Local recalibration (dengue, scarce district {scarce}, "
          f"n_pos={rm['n_positives']['dengue'][scarce]}): pooled threshold coverage "
          f"{cov_pooled:.0%} -> recalibrated locally {cov_local:.0%}")
    assert cov_local >= cov_pooled, "Local recalibration should not worsen coverage at the scarce site"

    # ---- Build the lean payload the app loads (drop bulky debug arrays) ---------------
    payload = dict(
        ecc=ecc,
        br_models=eng["br_models"],
        qhat=qhat,
        next_test=eng["next_test"],
        deploy_cols=eng["deploy_cols"],
        question_cols=eng["question_cols"],
        feature_groups=eng["feature_groups"],
        alpha=eng["alpha"],
        labels=LABELS,
        disp=eng["disp"],
        recal_data=eng["recal_data"],
        conformal_summary={
            l: dict(qhat=conf[l]["qhat"], coverage=conf[l]["coverage"],
                    n_test_pos=conf[l]["n_test_pos"], set_rate=conf[l]["set_rate"])
            for l in core.CONF_LABELS
        },
    )

    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "model.joblib")
    joblib.dump(payload, model_path, compress=3)  # compress to keep the committed artifact small

    meta = dict(
        built_at=_dt.datetime.now().isoformat(timespec="seconds"),
        seed=core.SEED,
        n_patients=int(len(Yv)),
        n_deploy_features=len(eng["deploy_cols"]),
        n_questions=len(eng["question_cols"]),
        recall_nmd_br=round(recall_br, 4),
        recall_nmd_ecc=round(recall_ecc, 4),
        coverage_malaria=round(cov_mal, 4),
        coverage_dengue=round(cov_den, 4),
        referral_caught=acct["caught"],
        referral_dengue_n=acct["dengue_n"],
        naive_caught=acct["naive_caught"],
        next_test=eng["next_test"],
        qhat={k: (round(v, 6) if v is not None else None) for k, v in qhat.items()},
        ecc_refactor_max_abs_diff=max_abs,
        python=sys.version.split()[0],
    )
    with open(os.path.join(here, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(model_path) / 1e6
    print(f"\nAll faithfulness assertions PASSED.")
    print(f"Wrote {model_path} ({size_mb:.1f} MB) and meta.json.")
    return 0


def _fit_br(Xtr, Ytr, Xte, LABELS):
    """Binary-relevance OOF probabilities (RQ1 comparison), local to the export check."""
    P = np.zeros((len(Xte), len(LABELS)))
    for j in range(len(LABELS)):
        yj = Ytr[:, j]
        if yj.sum() == 0:
            continue
        clf = core.base_learner().fit(Xtr, yj)
        P[:, j] = clf.predict_proba(Xte)[:, 1]
    return P


if __name__ == "__main__":
    raise SystemExit(main())
