# External Data — Acquisition & Reproduction Guide (SENTINEL · FIT 2026)

> **Compliance basis.** GuideBook FAQ #2: *"The use of external datasets is permitted only as supplementary data
> (e.g., for transfer learning or feature enrichment), provided they are open-source, legal, and clearly cited."*
> All three datasets below are **supplementary** to the official Mendeley `cf49v47z4c` cohort (which remains the
> primary, evaluated dataset). None are merged into the core training set. All are **open-source, publicly
> accessible, and CC-licensed**. Verified & exported on **2026-06-14**.

Anyone can reproduce every file in this folder by running the commands below verbatim. Requires `curl`, Python 3
with `pandas` (and `pyarrow`). No login/API key needed for any source.

---

## Summary of exported files

| File | Source dataset | Rows × Cols | License | Role in SENTINEL |
|---|---|---|---|---|
| `map_burkinafaso_pf_prevalence.csv` | Malaria Atlas Project — Pf parasite-rate surveys, Burkina Faso | 258 × 13 | CC-BY 4.0 | **Feature enrichment** (geo/temporal malaria prior) — Phase 6 |
| `zenodo_4121831_SISA_arboviral.csv` | SISA — clinical biomarkers, arboviral (Dengue/Chik), Brazil | 534 × 30 | CC-BY 4.0 | **Transfer-learning demo** (dengue symptom signature) — Phase 6 |
| `tanzania_fever_full.parquet` / `..._label_subset.csv` / `zenodo_166713_tanzania_fever.dta` | Tanzanian pediatric fever study | 1005 × 749 | CC-BY 4.0 | **Transfer / external-validation** (African pediatric febrile, malaria+typhoid labels) — Phase 6 |

> `_map_bf_raw.json` is the intermediate WFS GeoJSON (kept for traceability; not used directly).

---

## Dataset 1 — Malaria Atlas Project (MAP): Burkina Faso *P. falciparum* prevalence

- **Provider:** Malaria Atlas Project (MAP), University of Oxford / Telethon Kids Institute
- **Portal:** https://data.malariaatlas.org/ (Explorer: https://data.malariaatlas.org/explorer)
- **License:** CC-BY 4.0 (cite MAP)
- **What it is:** 258 georeferenced community parasite-rate surveys in Burkina Faso (year range **1985–2012**),
  with `examined`, `pf_pos`, `pf_pr` (parasite rate), lat/long, survey method. Median `pf_pr` ≈ **0.54** —
  confirms high-endemicity context of our cohort.
- **Why supplementary-safe:** aggregate, population-level, **no patient-level labels** → zero leakage risk.

### How to download (WFS — no key needed)
```bash
mkdir -p external_data

# 1. (optional) discover available layers
curl -sS "https://data.malariaatlas.org/geoserver/ows?service=WFS&version=2.0.0&request=GetCapabilities" \
  | grep -oiE "<Name>[^<]*</Name>"

# 2. fetch all Burkina Faso Pf survey points as GeoJSON
curl -sS "https://data.malariaatlas.org/geoserver/Explorer/ows?service=WFS&version=2.0.0&request=GetFeature&typeName=Explorer:public_pf_data&CQL_FILTER=country='Burkina%20Faso'&outputFormat=application/json" \
  -o external_data/_map_bf_raw.json
```
```python
# 3. flatten GeoJSON -> clean CSV
import json, csv
d = json.load(open('external_data/_map_bf_raw.json', encoding='utf-8'))
cols = ['site_name','latitude','longitude','rural_urban','year_start','year_end',
        'lower_age','upper_age','examined','pf_pos','pf_pr','method','country']
with open('external_data/map_burkinafaso_pf_prevalence.csv','w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for ft in d['features']:
        p = ft['properties']; w.writerow({c: p.get(c) for c in cols})
```
> Swap `Burkina%20Faso` in `CQL_FILTER` for any country. Use `Explorer:public_pv_data` for *P. vivax*.

### Citation
Malaria Atlas Project. *Global Pf Parasite Rate surveys* (Explorer `public_pf_data`). https://malariaatlas.org/ —
licensed CC-BY 4.0. (Underlying: Weiss et al., *The Lancet*, 2019; Pfeffer et al., *Malaria Journal*, 2018.)

---

## Dataset 2 — SISA: clinical biomarkers for arboviral infection (Dengue / Chikungunya)

- **Zenodo record:** https://zenodo.org/records/4121831  (DOI: 10.5281/zenodo.4121831)
- **License:** CC-BY 4.0
- **What it is:** 534 individuals (file is 534 rows × 30 cols) hospitalized with arboviral infection, with
  symptom columns (`SympHead`, `SympMuscle`, `SympRash`, `SympBleed`, `SympVomit`, `SympOrbital`, `Tourniquet`),
  history flags (`HistAllergies/Hyper/Asthma/Cancer/Diab`, `HistDeng`), and demographics. **Strong column overlap
  with our Burkina Faso symptom + history block** → ideal for a dengue-signature transfer-learning demo.
- **Associated paper:** SIMON analysis, doi:10.1101/2020.08.16.252767.

### How to download
```bash
curl -sS -L "https://zenodo.org/api/records/4121831/files/SISA.csv/content" \
  -o external_data/zenodo_4121831_SISA_arboviral.csv
```
```python
import pandas as pd
df = pd.read_csv('external_data/zenodo_4121831_SISA_arboviral.csv')
print(df.shape)  # (534, 30)
```

### Citation
SISA dataset, *Dataset of clinical biomarkers for prediction of the arboviral infection severity using SIMON
analysis*, Zenodo, 2020. https://doi.org/10.5281/zenodo.4121831 (CC-BY 4.0).

---

## Dataset 3 — Tanzanian pediatric fever study database

- **Zenodo record:** https://zenodo.org/records/166713  (DOI: 10.5281/zenodo.166713)
- **License:** CC-BY 4.0
- **What it is:** **1005 pediatric febrile patients × 749 columns** (Stata `.dta`). Contains malaria RDT &
  blood-smear (`rdtmalaria`, `bsmalaria`), typhoid RDT (`rdttyphoid`), symptom fields (`fever`, `headache`,
  `cough`, `vomiting`, `diarrhea`, `rash*`), vitals (`tempa`), and structured clinical diagnoses
  (`diagclin*`, `diaglist*`, `diaginv*`). **African + pediatric + malaria/typhoid labels** → closest distribution
  to our cohort; best candidate for external validation / transfer.

### How to download
```bash
curl -sS -L "https://zenodo.org/api/records/166713/files/feverstudyanalnew.dta/content" \
  -o external_data/zenodo_166713_tanzania_fever.dta
```
```python
import pandas as pd
df = pd.read_stata('external_data/zenodo_166713_tanzania_fever.dta', convert_categoricals=False)
print(df.shape)  # (1005, 749)
df.to_parquet('external_data/tanzania_fever_full.parquet')   # portable copy
```

### Citation
*Fever study database* (pediatric Tanzanian fever study), Zenodo, 2016. https://doi.org/10.5281/zenodo.166713
(CC-BY 4.0).

---

## Usage guardrails (do NOT violate — keeps us compliant + leak-free)

1. **Primary = official Mendeley cohort.** These three are **supplementary only** (feature enrichment + transfer).
   Never concatenate their rows into the core training set.
2. **Never** put external rows into the **conformal calibration set** (Phase 3) — it would void the coverage
   guarantee.
3. **Never** use them to inflate **fairness subgroup power** (Phase 5) — different population.
4. Different label taxonomies → any transfer use needs an explicit **label-mapping** to {Malaria, Dengue, Typhoid,
   Yellow fever}; report shift honestly (domain-similarity / covariate-shift analysis).
5. **Cite all three in the report** (Bab II / Bab IV) — required by FAQ #2 and helps the ≥3-journal-reference bar.

*Generated 2026-06-14 for SENTINEL / FIT Competition 2026, Track IV.*
