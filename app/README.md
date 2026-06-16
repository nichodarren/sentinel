# SENTINEL - triage prototype (deployment app)

🚀 **Live: [sentinel-fit.streamlit.app](https://sentinel-fit.streamlit.app)**

An uncertainty-aware **triage and referral assistant** for co-circulating vector-borne febrile
disease (malaria, dengue, typhoid, yellow fever), wrapping the research engine from the SENTINEL
notebook in a live Streamlit interface. It is the deployed (tier-one) successor to the inline
notebook prototype.

> **NOT FOR CLINICAL USE.** Research proof of concept on a single-region cohort of 299 patients.
> It supports prioritising scarce confirmatory testing; it does not diagnose.

## What it does

A clinician ticks the clinical signs a patient presents, and the assistant returns, live:

1. a **per-disease probability** from an ensemble of classifier chains (notebook Section 4);
2. a **per-label conformal prediction set** - the diseases that cannot be ruled out at 90%
   confidence, guaranteed only for malaria and dengue (Section 5.3);
3. an asymmetric **referral flag** that fires when dengue cannot be ruled out (Section 5.4);
4. the **value-of-information next test** that most resolves the residual uncertainty (Section 5.5);
5. a per-disease **"why" explanation** (SHAP over a per-disease model) and an **equity note**.

The model is **symptoms-only by design** - it must run with every laboratory field blank, the
surge condition it is built for - so entering a laboratory value does not change the score; the
assistant instead recommends which test would help most. Yellow fever and typhoid are shown for
context only and never drive a referral, because the cohort has too few cases to calibrate them.

Extras: a bilingual interface (English / French, since the cohort is francophone West Africa), a
compact 12-sign questionnaire (the Section 5.9 NSGA-II selection) or the full sign list, optional
age and sex, and illustrative example presentations.

## Architecture (and why the output is faithful to the notebook)

```
data.csv  ->  sentinel_core.build_engine()  ->  export_model.py  ->  model.joblib + meta.json
                       (frozen copy of the                                     |
                        notebook's methodology)                                v
                                                                          app.py (loads it)
```

- **`sentinel_core.py`** is a frozen, verbatim copy of the notebook's preprocessing, ensemble of
  classifier chains, per-label conformal calibration, referral rule, value-of-information ranking,
  and the 12-sign questionnaire. The notebook stays self-contained and does **not** import this.
- **`export_model.py`** builds the engine and **asserts faithfulness before saving**: that the
  refactored fast model reproduces the reference `fit_ecc` exactly, and that the headline numbers
  match the notebook (non-malarial dengue recall, conformal coverage, the 24/27-vs-8/27 referral
  result, the `wbc` next-test). If any assertion fails, `model.joblib` is not written.
- **`app.py`** loads `model.joblib` for instant start-up. If it is missing it rebuilds the engine
  in-session (cached), so the app still runs from nothing.

## Run locally

```bash
cd app
pip install -r requirements.txt
python export_model.py        # writes model.joblib + meta.json (asserts faithfulness); ~2 min
streamlit run app.py
```

`export_model.py` is optional if `model.joblib` is already present; the app will rebuild on first
run otherwise. `data.csv` must sit at the repository root (one level above this folder).

## Deploy to Streamlit Community Cloud

This app is already deployed at **[sentinel-fit.streamlit.app](https://sentinel-fit.streamlit.app)**.
To stand up your own instance:

1. Fork or push this repository to GitHub.
2. On https://share.streamlit.io create a new app pointing at **`app/app.py`** on the `main` branch.
3. Streamlit Cloud installs `app/requirements.txt` automatically. Because `model.joblib` is
   committed, the app starts immediately without retraining.

The same app can be hosted on a Hugging Face Space (Streamlit SDK) if preferred.

## Refresh the frozen model

Whenever the methodology in `sentinel_core.py` changes, rerun `python export_model.py` and commit
the regenerated `model.joblib` + `meta.json`. The assertions guard against shipping a stale or
drifted model.

## Files

| File | Role |
|---|---|
| `sentinel_core.py` | Frozen research engine (preprocessing + models + conformal + referral + VoI). |
| `export_model.py` | Builds the engine, asserts faithfulness, writes `model.joblib` + `meta.json`. |
| `app.py` | Streamlit UI. |
| `model.joblib` | Frozen full-cohort model + thresholds (loaded by the app). |
| `meta.json` | Provenance and the asserted headline metrics (shown in the model card). |
| `requirements.txt` | Pinned dependencies. |
| `.streamlit/config.toml` | App theme. |
