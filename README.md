# SENTINEL — Co-Infection-Aware Triage for Vector-Borne Febrile Disease

**FIT Competition 2026 · Track IV (Human Health / Digital Health Intelligence) · UKSW Salatiga**

🚀 **Live demo: [sentinel-fit.streamlit.app](https://sentinel-fit.streamlit.app)**

> When disasters such as floods trigger secondary surges of co-circulating vector-borne febrile
> disease, frontline health workers in low-resource settings must distinguish malaria, dengue,
> typhoid, and yellow fever despite overlapping symptoms and scarce confirmatory tests. SENTINEL is
> an uncertainty-aware **triage and referral assistant** that predicts a co-occurring disease profile
> from clinical symptoms alone, stays robust to missing laboratory tests, and quantifies per-label
> uncertainty so that dangerous cases can be referred safely.
>
> It is a research proof of concept, framework, and roadmap, **not a deployable clinical diagnostic.**

---

## Why this is different

Existing models force a single-disease label and depend on the same laboratory tests that go missing
during a surge, which is how they end up missing dengue and co-infections (reported sensitivity as low
as 0–5.5% in comparable cohorts). SENTINEL reframes the task:

- **Multi-label, not single-label.** It predicts the full co-occurring profile, so a patient who has
  both malaria and dengue is not collapsed into one winner. On the held-out non-malarial dengue cell
  (the hardest, n=14), the multi-label formulation recovers recall from the single-label failure to
  roughly 43%, the win coming from the *formulation* rather than from heavier modeling.
- **Symptoms-first, lab-free by design.** The model runs with every laboratory field blank, the exact
  surge condition it is built for. Missingness is treated as a feature, not an error.
- **It knows when it does not know.** Per-label conformal prediction yields calibrated prediction sets
  (coverage guaranteed for malaria and dengue), and an asymmetric referral rule fires whenever dengue
  cannot be ruled out, turning a moderate detector into a safe triage tool.
- **Honest about generalisation.** Performance is reported under leave-one-site-out validation, an
  equity audit across age, sex, and site, and a cross-cohort reproduction on independent Kenyan and
  Tanzanian data.

---

## What is in this repository

| Path | What it is |
|---|---|
| [`notebook.ipynb`](notebook.ipynb) | **The primary deliverable.** A single self-contained CRISP-DM notebook (162 cells, executes top to bottom with 0 errors, deterministic). |
| [`app/`](app/) | The **deployed Streamlit triage app** — a standalone service that wraps a frozen copy of the notebook's engine. See [`app/README.md`](app/README.md). |
| [`data.csv`](data.csv) | Primary cohort: Burkina Faso, 300 × 109 (Ouedraogo et al., 2025). |
| [`external_data/`](external_data/) | Supplementary cohorts used for independent reproduction and cross-cohort convergence only — never merged into training. Provenance in [`external_data/EXTERNAL_DATA_GUIDE.md`](external_data/EXTERNAL_DATA_GUIDE.md). |
| [`requirements.txt`](requirements.txt) | Pinned dependencies for the environment the notebook was validated on (Python 3.13). |

The notebook is organised explicitly around the six **CRISP-DM** phases:

1. **Business Understanding** (§1) — disaster framing, problem statement, ML goal, feasibility, related work
2. **Data Understanding** (§2) — detect-only exploratory analysis, data-quality verification, hypotheses
3. **Data Preparation** (§3) — leak-free cleaning, feature construction, site integration, readiness validation
4. **Modeling** (§4) — Binary Relevance to Ensemble of Classifier Chains over LightGBM (RQ1 + RQ2)
5. **Evaluation** (§5) — site-aware leave-one-site-out, Mondrian conformal prediction, the referral rule, value-of-information next-test ranking, an NSGA-II minimal-questionnaire designer, and a fairness audit
6. **Deployment** (§6) — an inline triage prototype and the multi-region generalisation roadmap

---

## The deployed app

The notebook deliberately stays self-contained. The interactive product lives separately in
[`app/`](app/) and is **live at [sentinel-fit.streamlit.app](https://sentinel-fit.streamlit.app)**.

A clinician ticks the signs a patient presents, and the assistant returns, live: a per-disease
probability, a conformal prediction set (the diseases that cannot be ruled out at 90% confidence), an
asymmetric referral flag, the value-of-information next test, a per-disease "why" explanation, and an
equity note. It offers a bilingual interface (English / French), a compact 12-sign questionnaire or
the full sign list, and a demonstration of local recalibration for a new site.

To run it locally:

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

See [`app/README.md`](app/README.md) for the architecture and the faithfulness guarantees that keep
the app's output identical to the notebook.

---

## Reproducing the notebook

Requires **Python 3.13**.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb --ExecutePreprocessor.timeout=300
```

The notebook reads only `data.csv` and the files under `external_data/`, all shipped here. It creates
its own scratch output directory on run, so a fresh clone reproduces every figure and number from
nothing. All matplotlib panels also render inline in the committed `.ipynb`, so they are visible
directly in the GitHub notebook viewer without re-running. A full cold execution takes on the order of
twenty minutes, the NSGA-II search in §5 being the bulk of it.

---

## Data and licensing

- **Primary cohort** (`data.csv`): Ouedraogo et al. (2025), Mendeley Data `cf49v47z4c`, Burkina Faso,
  **CC BY 4.0**.
- **External cohorts** (`external_data/`): all open-access and CC-licensed, used **only** for
  independent reproduction and cross-cohort convergence. External rows are **never** merged into model
  training, conformal calibration, or the fairness audit. The Kenya CSV is redistributed here for
  reproducibility; each file's upstream source, license, and acquisition command are recorded in
  [`external_data/EXTERNAL_DATA_GUIDE.md`](external_data/EXTERNAL_DATA_GUIDE.md) and cited in the
  notebook bibliography.

---

## Honesty guardrails

N is small (299 usable rows), single-region, pediatric-skewed, and class-imbalanced (yellow fever
n=12). The deliverable is therefore a **proof of concept, framework, and roadmap**, not a clinical
oracle. Conformal coverage guarantees are claimed only for malaria and dengue. The fairness analysis
is an audit, not a remedy. Active testing is one-step and population-level. Any real deployment carries
a "not for clinical use" disclaimer.
