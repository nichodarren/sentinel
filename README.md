# SENTINEL — Co-Infection-Aware Triage for Vector-Borne Febrile Disease

**FIT Competition 2026 · Track IV (Human Health / Digital Health Intelligence) · UKSW Salatiga**

> An uncertainty-aware **triage and referral assistant** that predicts a co-occurring vector-borne
> disease profile (malaria, dengue, typhoid, yellow fever) from clinical symptoms alone, robust to
> missing laboratory tests, and quantifies per-label uncertainty so that dangerous cases can be
> referred safely. It is a research prototype and framework, **not a deployable clinical diagnostic**.

---

## Status

| Component | State |
|---|---|
| `notebook.ipynb` (CRISP-DM deliverable) | **Sections 1–5 complete + §6.1.** 141 cells, executes top to bottom with 0 errors, deterministic. |
| Technical Report (PDF, 40% of score) | **Not in this repo yet** (authored separately). |
| Statement of Originality | **Not in this repo yet.** |
| Deployment §6.2 prototype (interactive `ipywidgets` T0) | Written as prose; interactive widget **not yet built**. |

The notebook follows the six **CRISP-DM** phases:

1. **Business Understanding** (§1) — problem framing, ML goal, feasibility, related work
2. **Data Understanding** (§2) — detect-only EDA, data-quality verification, exploratory hypotheses
3. **Data Preparation** (§3) — leak-free cleaning, feature construction, site integration, readiness validation
4. **Modeling** (§4) — Binary Relevance to Ensemble of Classifier Chains over LightGBM; RQ1 + RQ2
5. **Evaluation** (§5) — site-aware LOSO, Mondrian conformal prediction, referral rule, fairness audit
6. **Deployment** (§6) — triage artefact and multi-region generalisation roadmap

---

## Reproducing the notebook

Requires **Python 3.13**.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb --ExecutePreprocessor.timeout=300
```

The notebook is **self-contained**: it reads only the data files shipped in this repository and
writes its interactive-figure PNG fallbacks to `eda_figs/`. All matplotlib panels render inline in
the notebook, so they are visible directly on the GitHub notebook viewer without re-running.

### Build discipline (important for anyone continuing this work)

`notebook.ipynb` is **generated**, never hand-edited. To change it:

```bash
python build/build_notebook.py      # regenerates notebook.ipynb (unexecuted)
# then re-run the nbconvert --execute command above
```

`build/build_notebook.py` writes to the repository root automatically (no absolute paths), so it
works from any clone. Editing the `.ipynb` JSON by hand will be overwritten on the next build.

---

## Repository layout

```
.
├── notebook.ipynb              # THE deliverable (CRISP-DM, sections 1-5 + 6.1)
├── build/
│   └── build_notebook.py       # generator for notebook.ipynb (edit this, then rebuild)
├── data.csv                    # primary cohort: Burkina Faso, 300x109 (Ouedraogo et al. 2025)
├── external_data/              # supplementary cohorts (corroboration only, never merged into training)
│   ├── Western and coastal Kenya sick visit data 2019-2022/   # Kenya LaBeaud cohort (read at runtime)
│   ├── tanzania_fever_full.parquet                            # Tanzania cohort (read at runtime)
│   ├── zenodo_4121831_SISA_arboviral.csv                      # cited reference cohort
│   ├── map_burkinafaso_pf_prevalence.csv                      # cited geographic prior
│   └── EXTERNAL_DATA_GUIDE.md                                 # provenance + how each file was obtained
├── eda_figs/                   # PNG fallbacks for the interactive figures (regenerated on run)
├── requirements.txt            # pinned dependencies (the environment the notebook was validated on)
├── Dataset.pdf                 # track / dataset description
├── GuideBook.pdf               # competition rules
├── desciption.xlsx             # data dictionary (conceptual, not 1:1 to columns)
├── PLAN.md                     # master plan: 20 sections, RQs, roadmap, decisions (process documentation)
└── CLAUDE.md                   # session-continuity / onboarding notes (locked decisions, gotchas, state)
```

### For a teammate or AI continuing the work

Read **`CLAUDE.md` first** (orientation, locked decisions, current state, known gotchas), then
**`PLAN.md`** (the comprehensive master reference). Together they document the full reasoning behind
every modeling and framing decision, so the project can be picked up without re-deriving context.

---

## Data and licensing

- **Primary cohort** (`data.csv`): Ouedraogo et al. (2025), Mendeley Data `cf49v47z4c`, Burkina Faso, **CC BY 4.0**.
- **External cohorts** (`external_data/`): all open-access, CC-licensed, used **only** for independent
  reproduction and cross-cohort convergence. External rows are **never** merged into model training,
  conformal calibration, or the fairness audit. See `external_data/EXTERNAL_DATA_GUIDE.md` for the
  provenance and acquisition command of each file.

The Kenya CSV is redistributed here for reproducibility; its upstream source and license are recorded
in the guide and cited in the notebook bibliography.

---

## Honesty guardrails

N is small (299 usable rows), single-region, pediatric-skewed, and class-imbalanced (yellow fever
n=12). The deliverable is therefore a **proof-of-concept, framework, and roadmap**, not a clinical
oracle. Conformal coverage guarantees are claimed only for malaria and dengue. Fairness is an audit,
not a remedy. Any deployment would carry a "not for clinical use" disclaimer.
