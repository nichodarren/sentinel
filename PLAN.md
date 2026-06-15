# PROJECT PLAN — "SENTINEL"
### Symptom-based, co-infection-aware Engine for Non-laboratory Triage in Emergency Low-resource settings

> **Master reference document.** Single source of truth for what we build, why, in what order, and how we judge success. Everything (notebook, Technical Report, deployment) traces back here.
> **Status:** problem locked · reviewed by 4 subagents & corrected · entering Phase 1.
> **Build policy:** the project is a **full rebuild from scratch**. The teammate's `notebook.ipynb` (single-label) is **reference-only for what to avoid** — we do **NOT** adopt its approach or copy its code (we re-implement preprocessing from the data dictionary).

---

## 0. Document Meta

| Field | Value |
|---|---|
| **Competition** | FIT Competition 2026 — Data Science, **Track IV (Human Health / Digital Health Intelligence)** — UKSW Salatiga (international) |
| **Theme** | *Digital Impact for Humanitarian Response and Global Well-Being* |
| **Track IV Primary Focus (anchor)** | *"Utilizing bio-inspired systems and advanced analytics to mitigate the long-term secondary impacts of disasters on human well-being."* |
| **Dataset** | Mendeley *"Tabular dataset for AI-based Vector-Borne Disease prediction"* (DOI 10.17632/cf49v47z4c.2), Ouedraogo et al., 2025 — **300 patients × 109 attributes**, DO & DAFRA health districts, Hauts-Bassins, **Burkina Faso**; CC BY 4.0; **no prior ML publication (unbenchmarked)** |
| **Preliminary deadline** | **16 June 2026, 23:59 (WIB)** — submit `.ipynb` + Technical Report PDF (+ Statement of Originality) via Google Form |
| **Final round** | Top 5 → on-site UKSW, 8–9 July 2026 (12 h live coding + presentation) |
| **Working name** | **SENTINEL** (also evokes *sentinel surveillance*) |

### Table of Contents
1. Executive Summary · 2. Background & Motivation · 3. The Research Gap · 4. **Problem Statement & RQs** · 5. Novelty & Contributions · 6. Alignment with Primary Focus · 7. Dataset Profile · 8. Solution Architecture · 9. Methodology per RQ · 10. **Roadmap — Phases 0–8** · 11. **Phase 8 Deep-Dive + Evaluation** · 12. Evaluation Plan & Metrics · 13. Honesty Guardrails, Limitations & Ethics · 14. Deliverables ↔ Scoring Map (+ format checklist) · 15. Risk Register · 16. Timeline · 17. Tech Stack · 18. Reference Library · 19. Decisions / Next Steps · 20. Glossary

---

## 1. Executive Summary

When disasters (floods, cyclones, displacement) strike, they trigger **secondary surges of co-circulating vector-borne febrile diseases** — malaria, dengue, typhoid, yellow fever. Frontline workers must tell these apart, yet (a) the diseases present with **near-identical overlapping symptoms** and **can co-occur in one patient**, (b) **confirmatory lab tests are scarce/absent** exactly when demand peaks, and (c) existing ML tools **force a single-disease label** and **depend on the very labs that are missing** — so they systematically miss dengue and co-infections (documented sensitivity as low as **0–5.5%**) and fail silently.

**SENTINEL reframes the task** from "guess one disease" to **"predict the co-occurring disease profile from symptoms alone"** — robust to (and possibly informed by) missing labs, with **calibrated per-label uncertainty to safely refer dangerous cases**, a **"recommend the next test" module**, an **equity audit** across vulnerable subgroups, and an **honest path to multi-region generalization**. We build the **first ML benchmark** of an unbenchmarked Burkina Faso cohort, and (Phase 8) wrap it into a **prototype triage interface**.

**The one-paragraph pitch (for cover/abstract):**
> *We reframe vector-borne disease diagnosis in disaster-affected, lab-scarce settings from the prevailing single-label task into what it clinically is — a co-infection-aware, multi-label prediction problem solved from clinical symptoms alone, where the missingness pattern of laboratory tests is examined as a potential diagnostic-intent signal. Using ensemble classifier chains over missingness-native gradient-boosted trees, wrapped in per-label conformal prediction with cost-sensitive abstention and a value-of-information "recommend next test" module, and audited for fairness across children/adults/sex/site, SENTINEL turns laboratory scarcity from a weakness into the contribution — extending safe diagnostic reach where confirmatory testing collapses.*

---

## 2. Background & Motivation

**Vector-borne febrile disease is a mass global-health burden, concentrated in children and the tropics — and now globalizing.**

- **Malaria:** ~263–282 M cases and ~597,000 deaths in 2024 (WHO); African Region ~95% of cases/deaths; **children <5 ≈ 76% of deaths**. *Mirrors our data: 90% malaria, pediatric-heavy.*
- **Dengue:** record **14.6 M+ reported cases / 12,000+ deaths in 2024** across 100+ countries; ~half the world at risk. **Climate change is pushing *Aedes*/dengue into temperate zones** — autochthonous dengue now in **France, Italy, Spain, and the US (Florida, Texas, California)**. → **worldwide, not just tropical**.
- **Typhoid/enteric fever:** ~14.3 M cases / ~135,900 deaths (GBD 2017); the common **Widal test is so unreliable it is excluded from burden estimates** — vivid illustration of the diagnostic vacuum.
- **Yellow fever:** 84,000–170,000 severe cases / 29,000–60,000 deaths/yr in Africa; **13 (francophone-heavy) African countries** reported cases in 2023–24 — *our dataset's geography*.

**Disasters trigger these surges.** Floods/displacement → stagnant water + crowding + broken sanitation → malaria, dengue, typhoid, cholera. Concrete: 57,010 dengue cases after Brazil's 2008 floods; Pakistan's 2022 floods → documented malaria, dengue, typhoid, cholera epidemics. **This is the literal "secondary health impact of disasters" of the Track IV anchor.**

**The clinical crux:** all four present as **acute undifferentiated febrile illness (AUFI)**, so clinical-only diagnosis is error-prone, and **co-infection is dangerous** (pooled malaria–dengue ~4.2% in Africa, rising; severe-disease OR ≈ **3.94**). Dschang, Cameroon: **13.5% of malaria-negative fevers were dengue-positive** — routinely missed. The default "treat all as malaria" wastes drugs, delays care, and **fuels antimicrobial resistance** (70–88% of ACT-treated patients malaria-negative in some studies).

---

## 3. The Research Gap (prior art falls short)

| Prior approach | What they do | Why it fails our problem |
|---|---|---|
| Single-disease binary (Vietnam, Puerto Rico) | "dengue vs other", AUC 0.86–0.94 | Ignores other diseases & co-infection |
| Multiclass single-label (India, Kenya, Nigeria) | one disease per patient | Forces one answer; **multiclass accuracy 55–60%** vs 79–84% binary |
| Co-infection as a "combo class" | "both" = a 4th class | Can't generalize to unseen combos; fragments scarce samples |
| Rule-based CDS (IMCI → ALMANACH → e-POCT/e-POCT+) | guideline + POC biomarker | Not learned; **requires CRP/procalcitonin often unavailable**; single output |
| Consumer symptom-checkers | app triage | Diagnostic accuracy **19–37.9%**; not tropical-validated |

**The smoking gun (verified):** Mutai et al. (Kenya, **n = 6,208**, 6 ML algorithms) — only logistic regression detected any dengue at all, at **5.5% sensitivity**; the rest **0%**. *Caveat for honesty:* this failure is driven by the single-label framing **plus** low prevalence (~7.8%) and accuracy-tuned objectives — so multi-label framing is a *promising hypothesis*, not a proven cure. A 2025 BMJ Open review protocol confirms **no AI tool simultaneously diagnoses more than two diseases**. IMCI itself admits *"most children have more than one illness; a single diagnosis may not be possible."*

**White space (convergent across our research):** (1) genuine **multi-label co-infection** diagnosis for febrile illness; (2) **lab-free / missingness-aware** symptom-only prediction; (3) **"recommend next test"** value-of-information; (4) **francophone West-African** individual-level differential diagnosis (our cohort is **unbenchmarked**); (5) **disaster / secondary-impact** framing.

> ⚠️ A *single-label bio-inspired* approach (Antlion Optimization + RF/XGBoost) already exists for this disease family — so "bio-inspired feature selection" is **not** our novelty. Our novelty = the **co-infection multi-label + lab-free + safe-triage reframing and integration**.

---

## 4. Problem Statement & Research Questions

### 🎯 PROBLEM STATEMENT

> **When disasters trigger secondary surges of co-circulating vector-borne febrile diseases — malaria, dengue, typhoid, and yellow fever — frontline health workers in low-resource settings must distinguish and treat them despite three compounding obstacles: (1) near-identical, overlapping symptoms that can co-occur in the same patient; (2) confirmatory lab tests that are scarce, delayed, or unaffordable precisely when surge demand peaks; and (3) existing ML decision-support models that force a single-disease label and depend on those same missing lab inputs — causing them to systematically miss dengue and co-infections (sensitivity as low as 0–5.5%) and to fail silently rather than signal their uncertainty.**
>
> **Central question:** *How can we predict the complete co-occurring vector-borne disease profile of a febrile patient from clinical symptoms alone — robust to, and possibly informed by, missing laboratory data — while quantifying per-label uncertainty to safely refer dangerous cases, recommending the single most informative confirmatory test to run next, remaining equitable across vulnerable subgroups (children, sex, site), and offering an honest path to generalization beyond its single-region training data?*

### Research Questions

| RQ | Question | Primary success metric (re-anchored) |
|---|---|---|
| **RQ1 — Co-infection** | Can co-infection-aware **multi-label** learning recover minority-disease & co-infection detection where single-label fails? | **Recall on non-malarial dengue (n=14)** — the clinically decisive, hard cell — not the inflated "149 two-disease" figure. Report ECC vs Binary Relevance honestly (likely within CIs). |
| **RQ2 — Lab-free** | How much accuracy from **symptoms alone**, and does the **lab-missingness pattern as signal** help (with leakage controlled)? | Symptoms-only (honest ceiling) vs +mask vs +labs; **ablate confirmatory-test masks** (see §9). |
| **RQ3 — Safe triage** | Can **per-label calibrated/conformal uncertainty + cost-sensitive abstention** give safe referral for dangerous cases? | Per-disease coverage (Malaria/Dengue), dangerous-miss vs referral-rate curve; YF/Typhoid reported **descriptively**. |
| **RQ4 — Active testing** *(ext.)* | Which single confirmatory test maximizes diagnostic information per cost? | **Population-level** expected info gain, qualitative, **no performance claim**. |
| **RQ5 — Equity & transfer** *(ext.)* | Is performance equitable across well-powered subgroups, and what is the honest generalization roadmap? | Child-vs-adult & sex gaps for **Malaria/Dengue only**, min cell size n≥20, bootstrap CIs. |

**Scope (LOCKED):** RQ1–RQ3 = **headline backbone (must-ship)**. RQ4–RQ5 = **extensions / Final-round** (described in the report; implemented if core is solid).

---

## 5. Novelty & Claimed Contributions

1. **Conceptual reframing** — we **reframe** vector-borne febrile differential diagnosis (francophone West Africa) as a **genuine multi-label co-infection** problem (vs the prevailing single-label framing). *(Scoped claim — to our knowledge prior febrile-VBD models use single-label framing, but generic multi-label comorbidity methods exist; the novelty is the domain reframing + integration, not a new algorithm. Avoid the bare word "first" — say "to our knowledge" / "we reframe".)*
2. **Missingness-as-signal for VBD** — examining the **lab-test-ordering pattern as a diagnostic-intent feature**, enabling **lab-free triage** — with an **explicit leakage ablation** so the claim is honest.
3. **Safe multi-label triage** — **per-label conformal prediction with cost-sensitive abstention** that refers when a **dangerous miss** is plausible.
4. **Value-of-information "recommend next test"** — turning lab scarcity into actionable guidance (proof-of-concept).
5. **First publicly-documented (multi-label) benchmark** of the unbenchmarked Burkina Faso cohort; a francophone West-African differential-diagnosis reference. *(Scope the word "first": the Mendeley record carries no associated ML paper, and a teammate built a separate single-label notebook on the same cohort, so claim only "first publicly-documented / first multi-label", never a bare "first ML benchmark".)*
6. **Equity-first, disaster-framed positioning** with a credible **external-validation / federated-learning** roadmap countering "health data poverty."
7. **(Phase 8) A working prototype** embodying the framework as a usable triage interface.

> Publishable core = **#1 + #2 + #3** (reframing + integration). Everything else strengthens/extends it.

---

## 6. Alignment with Track IV Primary Focus

| Anchor phrase | How SENTINEL satisfies it |
|---|---|
| *"…secondary impacts of disasters…"* | VBD outbreaks = textbook secondary health impact of floods/displacement; SENTINEL preserves triage capacity when labs collapse. **Lead every report chapter with the flood→VBD-surge mechanism** (Pakistan 2022, Brazil 2008). Do NOT claim the model "predicts disaster impact." |
| *"…advanced analytics…"* | Multi-label classifier chains, missingness-native gradient boosting, per-label conformal prediction, value-of-information |
| *"…bio-inspired systems…"* | **Committed-minimal** module: **NSGA-II** multi-objective feature selection (objectives = performance ↑ / #questions ↓ / fairness-gap ↓) → Pareto front → designs the minimal triage questionnaire. Time-boxed; refit inside outer CV folds; benchmarked vs L1/MI. Distinct from the published single-label Antlion work. |
| Example 3: *"Algorithmic Fairness… equitable service access"* | Subgroup fairness audit (well-powered axes only) |
| *"…human well-being" / humanitarian-global* | Safe, equitable triage for under-served LMIC populations; global framing via climate-driven disease spread |

### 6.1 External supplementary datasets (acquired 2026-06-14 — **supplementary-only, GuideBook FAQ #2**)

> **Rule resolution (LOCKED):** GuideBook has two clauses — the main rules say *"not permitted to use any external datasets"*, but **FAQ #2 (lex specialis, wins)** permits external data *"only as supplementary data (e.g., for transfer learning or feature enrichment), provided they are open-source, legal, and clearly cited."* So external data is **allowed but strictly supplementary**: the official Mendeley cohort stays the **primary, evaluated** dataset; external rows are **never merged into core training**, never into conformal calibration, never used to inflate fairness power. All three below are open-source, CC-licensed, publicly downloadable with no key. **Verified & exported live on 2026-06-14** → `external_data/` (+ reproduction guide `external_data/EXTERNAL_DATA_GUIDE.md`).

| Dataset | Source / license | Size | Role | Phase |
|---|---|---|---|---|
| **Malaria Atlas Project — Pf parasite-rate, Burkina Faso** | MAP WFS GeoServer (`Explorer:public_pf_data`), CC-BY 4.0 | 258 georef surveys (1985–2012); median `pf_pr`≈0.54 | **Feature enrichment** — static geo/endemicity prior; grounds the disaster→VBD narrative with real numbers. Aggregate → **zero leakage risk** | 6 (opt. +prior ablation regime in 2) |
| **SISA — arboviral clinical biomarkers (Dengue/Chik)** | Zenodo 4121831, CC-BY 4.0 | 534×30; symptom cols (`SympHead/Muscle/Rash/Bleed/Vomit`, `Tourniquet`) + history flags — **near-mirror of our symptom/history block** | **Transfer-learning demo** — does the dengue symptom signature generalize cross-continent? | 6 |
| **Tanzanian pediatric fever study** | Zenodo 166713, CC-BY 4.0 | 1005×749; `rdtmalaria`/`bsmalaria`/`rdttyphoid` + symptoms — **African + pediatric + malaria/typhoid labels** | **External-validation / transfer** — closest distribution to our cohort | 6 |

*(IDAMS was probed but not found as a clean public file; SISA + Tanzania replace it with verified, column-matched cohorts. Kaggle "symptom" sets rejected — synthetic / unclear provenance.)*

> **⚠️ EXTERNAL-DATA PLAN OVERHAULED & IMPLEMENTED 2026-06-15 (supersedes the table above in part).** A from-scratch re-research (user-requested) replaced the primary external cohort. **Now used: the Kenya "Western and coastal Kenya sick visit data 2019-2022" cohort** (Kiener/LaBeaud/Ndenga/Mutuku, **Dryad DOI 10.5061/dryad.w9ghx3fxc**, CC, 1534×119; dengue confirmed by PCR `ufi_zcd_dengue_result` 30+, malaria by RDT 289+, plus clinician provisional-dx flags). It is the **live successor cohort of the same Stanford/Kenya group as Mutai et al. 2023** — the very "smoking gun" 0–5.5% paper cited in §3. The actual Mutai dataset (Dryad `rn8pk0pg1`) is **dead/withdrawn** (404 + 0 API versions). **Downloaded manually by the user** (Dryad file API requires a bearer token) to `external_data/Western and coastal Kenya sick visit data 2019-2022/`. **Role = ambitious but honest: reproduction of the single-label failure + cross-cohort convergence of findings, NOT granular model transfer** (its symptoms are pre-aggregated into 6 group flags, so no feature-for-feature transfer is possible — a limitation stated in-notebook and turned into the §6.2 design lesson). **Integrated into `notebook.ipynb` §4.4.1 / §5.2.1 / §6.2** (executed, 141 cells, 0 err): clinician dengue sensitivity **13.3%** + accuracy-tuned single-label ML **0.0%** (yet symptom AUC 0.73) reproduces the failure on a second African PCR-labelled cohort; the convergence table shows malaria moderately separable in every cohort while dengue/typhoid stay weak. **Tanzania D'Acremont (Zenodo 166713) RETAINED in a concise Mal/Typ-convergence role only** (malaria AUC 0.71 / typhoid 0.54 in §5.2.1; NOT a pillar). **SISA (4121831) + MAP DROPPED** (SISA has no in-file dengue label; MAP aggregate-prior marginal). Yellow fever has no public labelled external cohort → descriptive-only. Guardrails unchanged (primary cohort stays evaluated; external rows never in training/calibration/fairness; all cited). See memory `kenya-labeaud-external-dataset`.

---

## 7. Dataset Profile (verified against `data.csv`)

**Shape:** 300 patients × 109 columns. **Language:** French. **Load (critical):** `pd.read_csv('data.csv', sep=';', decimal=',', encoding='latin-1')` (use `cp1252` for cleaner accented headers; better: index/rename columns early — exact header strings are fragile).

**Feature groups:**
- **~79 binary symptom features** (OUI/NON) — nearly complete. *(Note: some lab/biomarker flags like thrombocytopenia/lymphocytopenia are OUI/NON and counted here; the line between "lab flag" and "numeric lab" is fuzzy.)*
- **~6 numeric laboratory variables** + ~9 lab flags — frequently missing.
- **Vital signs** (temperature, respiratory rate, pulse, blood pressure, capillary refill) — numeric, messy.
- **Test results** (TDR/rapid test, *goutte épaisse*) — `Positif`/`Négatif` (⚠️ **trailing whitespace** e.g. `'Négatif '`, plus NaNs — exact-match parsing will silently mis-encode; strip first).
- **9 medical-history flags** (diabetes, hypertension, sickle cell, rheumatic, autoimmune, osteoarthritis, allergies, cancer, asthma) — mostly empty.
- **Target:** 8 disease indicator columns (`Maladies diagnostiquées/<disease>`) + 1 free-text diagnosis.

**Target distribution (MULTI-LABEL):**

| Disease | Positive (of 300) | Use? |
|---|---|---|
| Malaria (Paludisme) | **270 (90%)** | ✅ |
| Others (catch-all) | 99 | ⚠️ NOT a target VBD → all-negative on the 4 targets (run a sensitivity check kept-vs-excluded) |
| Dengue | 56 | ✅ |
| Typhoid | 29 | ✅ (wide CIs; **27/29 co-occur with malaria** → nearly nested) |
| Yellow fever | 12 | ✅ (very wide CIs; descriptive only) |
| Chikungunya / Zika / Option 8 | 0 | ❌ drop |

**⚠️ Two label geometries — don't confuse them:**
- **Including "Others"** (as in raw counts): labels/patient 0=1, 1=141, **2=149**, 3=9. *Most of the "149 two-disease" is "Malaria + Others (catch-all)", NOT true co-infection.*
- **Core TARGET geometry (the 4 diseases the model predicts, Others excluded):** labels/patient **0=15, 1=204, 2=80, 3=1**; Malaria-only 193, Malaria+Dengue 41. **Non-malarial dengue n=14, non-malarial YF n=3, non-malarial typhoid n=2** — these tiny cells are the real difficulty and the honest RQ1 yardstick.

**Demographics:** Age 0.1–85 yr, **median ≈12.5** (≈13.5 only if age-fractions are dropped — be consistent with the cleaning rule); gender **152 F / 146 M (2 missing)**.

**Missingness (top):** MUA 92% · capillary refill 70% · **free-text "other symptoms presented" 69%** *(this is the free-text field col 107 — NOT the "Others" diagnosis target, which is only 0.3% missing)* · blood pressure 57% · lymphocytes 51% · hematocrit 35% · weight 35%. **91/109 columns are <5% missing.** *Symptoms complete, labs sparse — the canonical low-resource signature (core of RQ2).*

**Data-quality hazards to FIX:** entry errors (temperature **3909 °C**, pulse **124000**, BP **"10976"**); mixed BP separators (`120/80`, `104|58`); age fractions (`7/12`=7 months, `15/12`); duplicate columns (headache ×2, "frequent urination" ×2); French decimals; **1 row is all-NaN across every disease column** (counted as "0 diseases" but is actually *missing labels* → decide **drop vs impute** up front, plus the 2 missing-gender / 2 missing-age rows). Encode OUI/NON→1/0, Positif/Négatif→1/0 (strip whitespace), Homme/Femme→1/0.

---

## 8. Solution Architecture (SENTINEL)

```
   Patient (febrile) ─▶ INPUT: ~79 symptoms (+ optional labs, + missingness mask)
                                  │
                    PREPROCESSING (leak-free, per-fold)  ── full rebuild from data dictionary
                    clean FR data · encode · BP split · age-fix · build missingness-mask
                                  │
                    CORE: co-infection-aware MULTI-LABEL                     ◀ RQ1, RQ2
                    Ensemble Classifier Chains (sklearn.multioutput) over
                    missingness-native gradient-boosted trees → per-disease P(Mal/Den/Typ/YF)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                          ▼
  SAFETY (RQ3)            NEXT-TEST (RQ4)            EQUITY AUDIT (RQ5)
  per-label conformal     population-level VoI:      well-powered subgroups:
  set + cost-sensitive    "run RDT / FBC?"           child/adult, sex (Mal/Den)
  abstain/refer           (proof-of-concept)
        └─────────────────────────┼─────────────────────────┘
                                  ▼
   OUTPUT: disease profile + per-label confidence + referral flag + next-test rec + explanation  ◀ Phase 8 UI
```

---

## 9. Methodology per Research Question

**RQ1 — Co-infection multi-label.**
- **Binary Relevance** (baseline) → **Ensemble of Classifier Chains (ECC)** via **`sklearn.multioutput.ClassifierChain`** (multiple random chain orders, pass **predicted probabilities** down the chain, average). **Do NOT use `scikit-multilearn`** (unmaintained, breaks on sklearn 1.6) and **NOT label-powerset** (too many sparse combos).
- Base learner: **LightGBM** (native NaN handling). Drop Chikungunya/Zika; model {Malaria, Dengue, Typhoid, Yellow fever}.
- Baselines to beat: prevalence, **single-label multiclass (reproduce the failure mode — high-value, low-cost)**, Binary Relevance.
- **Honest framing:** Typhoid is nearly nested in malaria (chain learns it trivially); the real test is **non-malarial dengue recall (n=14)**. Pre-register that ECC-vs-BR may be **within CI**.

**RQ2 — Lab-free + missingness-as-signal (with leakage control).**
- Build a **missingness mask** (one indicator per lab); three regimes head-to-head: **(i) symptoms only (honest lab-free ceiling) · (ii) symptoms + mask · (iii) symptoms + imputed labs**.
- **Leakage ablation (mandatory):** separate the mask of **confirmatory tests** (TDR, thick smear, dengue RDT) from **generic labs** (FBC, CRP). If the gain concentrates in confirmatory-test masks, that is **label leakage (clinician intent), not lab-free triage** — report it as such. Frame the mask as a hypothesis with a **stated confound**.
- Imputation only for regime (iii): IterativeImputer + add-indicator, **inside each fold**. **No SMOTE** (over-engineers at N=300).

**RQ3 — Safe triage (corrected).**
- **Per-label (Mondrian) split-conformal, hand-rolled** (~20 lines: sort per-disease nonconformity scores, take 1−α quantile) — so YF's tiny set doesn't pollute Malaria's. Use MAPIE only for a single-label baseline if desired. **Do not claim joint per-patient set coverage** (multi-label conformal does not provide it as such).
- **Restrict calibrated coverage guarantees to Malaria & Dengue** (enough n). For **YF (12) / Typhoid (29)**: report abstention/referral **descriptively** with "coverage not calibratable at this n." Report **achievable discrete coverage levels**, not a nominal 90%. Consider **cross-conformal / jackknife+** to reuse all rows.
- **Cost-sensitive abstention/referral:** asymmetric costs (missing severe malaria / dengue hemorrhagic / YF ≫ false alarm) → referral policy; report **dangerous-miss rate vs referral-rate** curve.

**RQ4 — Active testing (extension).** Myopic value-of-information at **population level** (expected info gain per candidate test), **qualitative, no performance claim** (per-patient conditionals are un-estimable at N=300; also circular with the missingness signal — state both limits).

**RQ5 — Equity & transfer (extension).** **Audit only.** Pre-specify **minimum cell size n≥20 with positives**; audit only **well-powered axes** (child-vs-adult, sex) and only **Malaria/Dengue**; bootstrap CIs; "no detected gap ≠ no gap" (power caveat). NSGA-II feature selection: **refit inside each outer CV fold**, report **feature-set stability (Jaccard)** as the headline rather than one "optimal" set. Generalization roadmap: pre-registered geographical external-validation + federated/few-shot-transfer (data never leaves source country); a **cited supplementary external cohort** is permitted (GuideBook FAQ#2) to make this concrete.

**Validation protocol (whole project) — corrected.**
- **Multi-label stratification** via iterative stratification; given YF=12, **drop to 3-fold** for rare-label metrics OR (preferred) **pool out-of-fold predictions across all 300 rows, then compute rare-class metrics once with bootstrap CIs** (far more stable than averaging fold-wise undefined values). Avoid nested 5-fold on rare labels.
- Primary metric: **macro-F1 + per-disease recall**; guardrails: PR-AUC (OVR), balanced accuracy, **calibration (ECE)**. All numbers with **bootstrap CIs**; flag any label absent from a fold. **Everything leak-free, per-fold (sklearn Pipeline).**

---

## 10. Roadmap — Phases 0–8

| Phase | Name | Goal | Maps to |
|---|---|---|---|
| **0** | Lock & Audit | Freeze problem; full data audit (this PLAN.md) | Report Ch. I, III |
| **1** | **Reframe & Clean** | Convert to multi-label; **leak-free preprocessing rebuilt from scratch (data dictionary; NOT the teammate's code)**; build missingness mask; rich EDA | Notebook viz 20% + preproc 20% |
| **2** | Core Modeling | ECC via `sklearn.multioutput`; symptoms-only vs +mask vs +labs; reproduce single-label failure | **RQ1, RQ2** · model-perf 20% |
| **3** | Safety Layer | Per-label conformal + cost-sensitive abstention | **RQ3** |
| **4** | Active Testing *(ext.)* | Population-level "recommend next test" | RQ4 |
| **5** | Equity *(audit)* + **bio-inspired (committed-minimal)** | Subgroup audit; NSGA-II minimal-questionnaire + Pareto-front viz (time-boxed, in-fold) | RQ5 + Primary-Focus "bio-inspired" |
| **6** | Global Framing | External-validation/federated roadmap; WHO + FUTURE-AI alignment. **NOW CONCRETE:** run transfer/few-shot demo on real external cohorts (SISA dengue, Tanzania pediatric fever) + domain-similarity (covariate-shift) analysis; +MAP geo-prior feature-enrichment. All **supplementary-only** (see §6.1), datasets already exported to `external_data/` | Report Ch. IV–V |
| **7** | Polish | Clean reproducible notebook + Technical Report (≤20 pp, English, format-compliant) | **Graded deliverables** |
| **8** | Deployment + Eval | Prototype triage interface (T0 inline → T1 deployed) + its evaluation | Differentiator + Final head-start |

> **CRITICAL PATH (must-ship for submission):** Phases **1–3** (core + safety) + Phase **7** (notebook + report) + Phase **8 T0** (inline ipywidgets demo). 
> **Pre-committed to Final-round (NOT submission-blocking):** RQ4, full RQ5, full NSGA-II, Phase 8 **T1/T2** deployed app. Treat these as droppable by default, not as a fallback.
> **Sequencing rule:** core first, skin second. A polished UI on a weak model is worse than a strong model with a minimal UI.

> **⚠️ FINAL DELIVERABLE STRUCTURE — `notebook.ipynb` follows CRISP-DM (decided 2026-06-14).** The single graded notebook to be submitted is **`notebook.ipynb`**, organised by the six CRISP-DM phases (1 Business Understanding · 2 Data Understanding · 3 Data Preparation · 4 Modeling · 5 Evaluation · 6 Deployment). The phase **structure** = CRISP-DM; the phase **content** = this roadmap. So the roadmap phases above now map onto CRISP-DM sections, NOT onto separate notebooks: roadmap Phase 1 → CRISP-DM Sections 2–3; Phase 2 → Section 4; Phase 3 + RQ5 → Section 5; Phase 8 → Section 6. **`notebook.ipynb` must be fully self-contained** (it is submitted alone — NO references to PLAN.md, the report, `sentinel.ipynb`, or any other file). Built via `d:/tmp/build_notebook.py` (edit builder → rebuild → `nbconvert --execute`; never hand-edit). Status 2026-06-15: **Sections 1+2+3+4 done** (**111 cells**, 0 errors, 16/16 readiness PASS, 10 PNG figures, 0 em/en dash, +`age_band` → 90 features; **§2 carries 8 static + 2 Plotly interactive figures** — incl. V1 UpSet §2.3, interactive missingness §2.5.2, interactive symptom-network §2.5.5). **§4 Modeling (= roadmap Phase 2, RQ1+RQ2)** ports/extends the plan: 4.1 technique (BR→ECC, LightGBM, malaria-first), 4.2 dual CV (`iterstrat.MultilabelStratifiedKFold` pooled-OOF + `GroupKFold`-by-site, leak-free in-fold pipeline), 4.3 baselines (prevalence + single-label failure), 4.4 RQ1, 4.5 RQ2 (3 regimes + mask ablation 2-panel fig), 4.6 summary. **NEW dependency `iterstrat`** (pip-installed). **Site-cliff/LOSO + conformal + fairness + clustering + V6 phenotype map deferred to §5** per the §2.6 hypothesis contract. `sentinel.ipynb`/`sentinel_eda.ipynb` are retained as the evidence-first working notebooks that `notebook.ipynb` ports from; whether they ALSO ship is a Phase-7 decision (TBD).

> **📊 §4 RESULTS (executed, pooled-OOF, SEED=42).** RQ1: single-label reproduces the documented failure (non-malarial dengue recall **21.4%**, all 81 co-infected patients lose their 2nd label); multi-label recovers it — **Binary Relevance NMD recall 42.9% [21.4, 71.4]; ECC 35.7% [14.3, 64.3]** (overlap in CI → the WIN is the multi-label *formulation*, not ECC-over-BR; ECC carried forward for §5 conformal). Per-disease ECC symptoms-only: malaria F1 0.92 / PR-AUC 0.94, dengue F1 0.33, YF 0.12, typhoid 0.28; macro-F1 0.41 (dragged by YF n=12 & typhoid n=29 statistical floors). RQ2: mask ablation is a **near-null** — confirmatory-test mask shows **no leakage**, generic mask marginal+mixed, NMD recall flat at 0.357 across all mask variants; only `+labs` moves a cell (malaria recall 0.93→0.98) and it does **not** help dengue. ⚠️ These are within-population (optimistic); the honest cross-site **LOSO** numbers come in §5. See **§5a positioning** below.

> **🧭 §5a DEPLOYMENT POSITIONING — LOCKED 2026-06-15: the deliverable is an UNCERTAINTY-AWARE TRIAGE/REFERRAL ASSISTANT, not a detector/diagnostic.** User raised that the §4 metrics look weak for a *detection* task; resolved by re-scoping (consistent with the problem statement, which always said triage/refer/recommend-test, never diagnose). Moderate dengue recall is **reframed as a referral trigger**: a model that outputs "cannot rule out dengue → refer / request confirmatory test" is clinically safer than an over-confident detector that silently misses dengue. Justification stack for "low" metrics: (1) comparison-framing vs single-label (0–5.5% lit. → ~43% here); (2) moderate dengue is a *finding* consistent with the EDA (no single symptom separates dengue, J=0.32; high dengue F1 would imply leakage); (3) honesty + conformal "knows when it doesn't know" = clinical value beyond raw recall. This makes **§5 (conformal + LOSO) the decisive, non-optional section** that converts moderate recall into safe triage. Guardrails unchanged (conformal coverage only Malaria/Dengue; "next test" population-level qualitative). Detailed rationale: memory `deployment-positioning-triage`.

---

## 11. Phase 8 Deep-Dive: Deployment + Strategic Evaluation

### 11.1 What it is
Wrap the trained framework in a **lightweight, single-purpose, offline-capable prototype** (T0: `ipywidgets` **inline in the notebook — already installed, zero deploy risk**; T1: Gradio/Streamlit on free hosting — *gradio not yet installed; defer install to post-submission*). The clinician/CHW ticks **symptoms** (+ optional labs) and receives: multi-label disease profile + **per-label confidence set**, **safety/referral flag**, **"recommend next test"**, a plain-language **explanation** (top symptoms via SHAP), and an equity/uncertainty caveat + **"not for clinical use" disclaimer**.

### 11.2 Strategic evaluation — verdict: DO IT, tiered & honest
**For:** embodies the humanitarian theme; closes the research→artifact loop; memorable + lifts report "contribution" (10%); **front-loads the Final** (presentation 60%); forces output-design discipline. **Against:** not directly scored in preliminary; time risk (~3 days); over-claiming a clinical product (N=300 is proof-of-concept); scope creep.
**Tiers:** **T0** inline ipywidgets demo (must land before report is finalized, for screenshots) → **T1** deployed app (around/after submission; the Final weapon) → **T2** polished app (stretch). **Frame as a prototype** with a visible non-clinical disclaimer. *Final-round note: notebook weighting shifts to Viz 10 / Preproc 10 / Model 20 / Presentation 60 — the deployment is your presentation weapon.*

### 11.3 Success criteria & evaluation methods
Functional (end-to-end works) · **Faithfulness** (app output == notebook model output) · **Lab-free robustness** (works with all labs blank) · Latency (<1–2 s) · Interpretability (shows *why*) · **Safety behavior** (abstains/refers on dangerous+uncertain edge cases) · Usability (cognitive walkthrough, proxy CHW) · Equity surface (caveat visible) · Honesty (disclaimer present). **Out of scope (future work):** real clinical/field study, prospective validation, regulatory clearance.

---

## 12. Evaluation Plan & Metrics

- **Primary:** macro-F1 + **per-disease recall** (esp. **non-malarial dengue**). **Secondary:** PR/ROC-AUC (OVR), balanced accuracy, weighted-F1, **ECE**. **Multi-label:** subset accuracy, Hamming loss, label-ranking AP. **Safety:** dangerous-miss vs referral-rate; conformal coverage vs set size (per label). **Fairness:** well-powered subgroup gaps with bootstrap CIs.
- **Validation:** iterative-stratified K-fold; **pooled out-of-fold predictions + bootstrap CIs** for rare classes (not nested 5-fold). Report uncertainty everywhere.
- **Baselines:** prevalence; **single-label multiclass (reproduce 0–5.5% failure)**; Binary Relevance; +labs vs symptoms-only.

---

## 13. Honesty Guardrails, Limitations & Ethics

- **N = 300, single region, pediatric-skewed, severely imbalanced** (YF 12; Chik/Zika 0) → **proof-of-concept + framework + roadmap, NOT a deployable clinical diagnostic.**
- **Conformal:** guarantees only for Malaria/Dengue; YF/Typhoid descriptive; no joint per-patient coverage claim.
- **Missingness-as-signal:** report the leakage ablation; give symptoms-only-without-mask as the honest lab-free ceiling.
- **Fairness = audit only** (too small to mitigate); active testing = **one-step, population-level**; NSGA-II refit in-fold + stability reporting.
- **Novelty scoped** to "vector-borne febrile differential diagnosis, francophone West Africa"; multi-label = promising hypothesis, not proven cure for the Kenya failure.
- Position as **triage / risk-stratification**, not a "diagnostic oracle."
- **Deployment = prototype** with a visible "not for clinical use; requires external validation" disclaimer.
- **External data:** the official 300-patient set is the core; **supplementary open-source, legal, cited external data IS permitted** (GuideBook FAQ#2) — use it (if at all) only to strengthen the generalization narrative, with citation. **Acquired & exported 2026-06-14** (see §6.1 + `external_data/`): MAP Burkina-Faso prevalence (enrichment prior), SISA arboviral & Tanzania pediatric-fever cohorts (transfer/external-validation). **Hard guardrails:** never merge external rows into core training, conformal calibration, or fairness-power; transfer needs explicit label-mapping to {Mal,Den,Typ,YF} + honest domain-shift reporting; cite all three (Bab II/IV — also helps the ≥3-journal-ref bar).
- **No over-engineering** (the teammate-notebook failure mode): no SMOTE, no heavy stacking/HP-search at N=300; novelty from reframing + integration.
- **Ethics anchors:** WHO *Ethics & Governance of AI for Health* (2021; 2024 LMM update); **FUTURE-AI** (BMJ 2025); **Health Data Poverty** (Lancet Digital Health 2021) — building on under-represented LMIC data reduces inequity *if* data sovereignty is respected.

---

## 14. Deliverables ↔ Scoring Rubric Map (Preliminary)

| Deliverable | Rubric component | Weight | Phases |
|---|---|---|---|
| **Notebook (.ipynb)** | Visualization & understanding | **20%** | Phase 1 (EDA) — *budget equal effort here; high deterministic payoff* |
| | Appropriateness of preprocessing | **20%** | Phase 1 (Issue→Fix register as viz) |
| | Model performance / evaluation | **20%** | Phases 2–3 (+ honest CV) |
| **Technical Report (PDF)** | Intro / problem / solution / contribution | 10% | §1–§6 |
| | Literature Review | 10% | §2–§3 + Reference Library |
| | Methodology | 10% | §7–§9 |
| | Results & Discussion | 10% | Phases 2–5 + Phase 8 |
| **Statement of Originality** | administrative gate | — | committee template |

> **Highest-leverage insight (reviewer-convergent):** at N=300 the model ceiling is limited and judges know it; **Viz 20% + Preprocessing 20% = a third of the whole preliminary score and are scored deterministically.** Make the data-quality Issue→Fix register and missingness-pattern analysis into rich narrated visualizations (these *also* supply the empirical evidence for "missingness-as-signal").

**📋 Report Format Compliance Checklist (admin-DQ gate — bake into Phase 7 from the start):**
- [ ] A4 · Times New Roman 12 · line spacing 1.15–1.5 · margins **T 3 / B 3 / L 4 / R 3 cm**
- [ ] Front matter (cover, abstract, ToC) in **lowercase Roman numerals** (bottom-right); body **Arabic numerals** (bottom-right)
- [ ] **≤ 20 pages** (excl. cover, ToC, content?, appendices) · **English only**
- [ ] Mandated structure **Bab I–V** (Intro · Lit Review · Methodology · Results & Discussion · Conclusion) + References + Appendix
- [ ] File name **`UNIVERSITY NAME_TEAM NAME`**
- [ ] **≥ 3 peer-reviewed journal references** in Bab II *(significant scoring factor even at Preliminary — FAQ#7; aim 6–8; re-verify the ⚠️ sources)*
- [ ] Statement of Originality (from committee link) attached · submitted from **registered email** via Google Form

---

## 15. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Over-engineering at N=300 (overfit) | High | High | Pooled-OOF + CIs; simple ECC; no SMOTE/heavy stacking |
| Rare classes (YF 12, Typh 29) unstable | High | Med | Wide CIs; descriptive only; non-malarial-dengue as RQ1 yardstick |
| **`scikit-multilearn` breakage** | Was High | High | **Resolved — use `sklearn.multioutput` + hand-rolled ECC/conformal** |
| Scope too large for ~3 days | High | High | Hard critical path (RQ1–3 + report + T0); extensions pre-committed to Final |
| Under-investing Viz/Preprocessing (40%) | Med | High | Equal time budget; Issue→Fix + missingness as flagship viz |
| Conformal sets too wide (small calib) | Med | Med | Per-label; report honestly; cross-conformal |
| Missingness-as-signal = leakage | Med | Med | Confirmatory-test ablation; report honest ceiling |
| Disaster-bridge judged a stretch | Med | High | Lead each chapter with flood→VBD mechanism + citations |
| "bio-inspired" box unticked | Was Med | Med | **Resolved — committed-minimal NSGA-II (in-fold, time-boxed) + Pareto viz** |
| Citation accuracy (6 ⚠️ paywalled) | Med | Med | Re-verify all ⚠️ sources before submit |
| **Submission ops** (wrong email/format/naming) | Low | High | Checklist: registered email, monitor WhatsApp group, `UNIVERSITY_TEAM` naming, Statement of Originality, both files on Google Form |
| Originality | — | — | **Resolved — teammate is registered; additionally we full-rebuild our own code, so no third-party-code exposure** |

---

## 16. Timeline (to 16 June 2026)

- **Day 1:** Phase 1 (rebuild clean + multi-label reframe + **rich EDA**) → start Phase 2.
- **Day 2:** Finish Phase 2 (core: symptoms-only vs +mask vs +labs; reproduce single-label failure) → Phase 3 (per-label conformal + abstention). Begin report Bab I–III.
- **Day 3:** (time permitting) Phase 5 bio-inspired Pareto viz; **Phase 8 T0** inline demo → finalize report Bab IV–V + notebook polish → **submit** (run the §14 checklist).
- **Post-finalist (23 Jun–7 Jul):** RQ4, full RQ5, **Phase 8 T1/T2** deployed app, ≥3 journal refs, PPT.

---

## 17. Tech Stack

- **Data/ML:** Python, pandas, numpy, scikit-learn (**`multioutput.ClassifierChain` / `MultiOutputClassifier`** — *not* scikit-multilearn), LightGBM, (CatBoost optional — prefer single family to avoid over-engineering).
- **Uncertainty:** **hand-rolled per-label split/cross-conformal**; calibration (isotonic/Platt). MAPIE only for single-label baseline if needed.
- **Explainability:** SHAP.
- **Bio-inspired (committed-minimal):** pymoo (NSGA-II) — verified installs clean on Py3.13.
- **Deployment:** **ipywidgets (T0, installed)**; Gradio/Streamlit (T1, install post-submission); joblib.
- **Repro:** fixed seeds; pinned env; clean top-to-bottom notebook; **all transforms inside CV folds**.

---

## 18. Reference Library (verified; ⚠️ = re-verify citation details — publisher 403'd the automated fetch)

**Clinical/epi:** WHO *World Malaria Report 2024* · WHO *Dengue 2024 update* (WER) + fact sheet · Messina et al., *Global distribution & risk of dengue*, **Nature Microbiology** 2019 · GBD *Typhoid & paratyphoid*, **Lancet Infect Dis** 2019 · WHO *Yellow fever* (2024) · *Outbreaks after floods*, PMC 2024 · WHO *Communicable diseases following natural disasters* 2006 · *Malaria–dengue co-infection in Africa* (OR≈3.94), **Malaria Journal** 2023 · *Arboviruses in Dschang, Cameroon*, **PLOS NTD** 2022 · *AUFI aetiologies S/SE Asia*, **PLOS NTD** 2019.
**ML/methods:** **Mutai et al., Kenya (0–5.5%)**, *PLOS Glob Public Health* 2023 (PMC10370704) · *AI differentiating tropical infections* (55–60% vs 79–84%), **PLOS NTD** 2022 · *Tropical-fever AI review protocol* ("no tool diagnoses >2 diseases"), **BMJ Open** 2025 ⚠️ · Read et al., *Classifier chains*, **Machine Learning** 2011 · *Multi-label comorbidity identification*, **Comput Biol Med** 2017 · Lipton et al., *Modeling missing data* ("tests run can be as predictive as results"), **MLHC** 2016 · *MoDN modular CDS*, medRxiv 2022 ⚠️ · e-POCT, **PLOS Medicine** 2017 / DYNAMIC e-POCT+, **Nature Medicine** 2023 · *Conformal cost-aware deferral for triage*, **Scientific Reports** 2026 ⚠️ (confirm in-press) · *Symptom-checker accuracy*, **npj Digital Medicine** 2022.
**Governance/equity:** WHO *Ethics & Governance of AI for Health* 2021 + LMM 2024 · *FUTURE-AI*, **BMJ** 2025 · Ibrahim et al., *Health data poverty*, **Lancet Digital Health** 2021 ⚠️ · *Limits of fair medical-imaging AI*, **Nature Medicine** 2024 · *Globalizing fairness (AI/colonialism/Africa)*, arXiv 2024.
**Dataset & prior art:** Ouedraogo et al., **Mendeley** cf49v47z4c, 2025 (our dataset) · *Hybrid ML (Antlion + RF/XGBoost) for VBD* — the single-label bio-inspired work to differentiate from ⚠️.
**External supplementary (CC-BY, cite in Bab II/IV — see §6.1):** Malaria Atlas Project, *Global Pf parasite-rate surveys* (Weiss et al., **Lancet** 2019; Pfeffer et al., **Malaria Journal** 2018) · SISA arboviral dataset, **Zenodo** 10.5281/zenodo.4121831, 2020 · *Tanzanian pediatric fever study database*, **Zenodo** 10.5281/zenodo.166713, 2016.
*(6 sources flagged ⚠️ — re-verify before final submission.)*

---

## 19. Decisions / Next Steps

**Decisions — ALL LOCKED:**
1. ✅ **Problem Statement + name "SENTINEL"** — approved.
2. ✅ **Scope:** RQ1–RQ3 = headline backbone; RQ4–RQ5 = extensions/Final.
3. ✅ **Phase 8:** T0 (inline demo) for submission, T1 (deployed) for Final; framed as prototype.
4. ✅ **Bio-inspired = committed-minimal** — NSGA-II minimal-questionnaire designer (in-fold, time-boxed, benchmarked vs L1/MI; Pareto-front viz). Degrades to future-work only if Day-3 core is genuinely at risk.
5. ✅ **Build policy:** **full rebuild from scratch.** ⚠️ **NAME-CLASH NOTE (2026-06-15):** the filename `notebook.ipynb` now denotes **OUR CRISP-DM graded deliverable** (the from-scratch multi-label rebuild), **NOT** the teammate's notebook. The teammate (a registered team member) authored a *separate* single-label notebook that is **not present on disk** in this workspace; it remains a **concept-only** reference for what to avoid (single-label label-powerset, SMOTE, over-engineering). We do not adopt its approach or copy its code; we re-implement all preprocessing from the data dictionary. Reading this decision literally as "the submitted `notebook.ipynb` is the teammate's avoided notebook" would be **wrong** — the submitted file is our original rebuild. (CLAUDE.md carries the same clarification.)
6. ✅ **External data (2026-06-14, REVISED & IMPLEMENTED 2026-06-15):** **supplementary-only** strategy adopted (GuideBook FAQ #2 wins over the blanket "no external" clause). **The plan was overhauled on 2026-06-15** after a from-scratch re-research: the **Kenya LaBeaud sick-visit cohort** (Dryad `w9ghx3fxc`, PCR dengue + RDT malaria; live successor of the Mutai "smoking gun" group) is the chosen external dataset, integrated into `notebook.ipynb` §4.4.1/§5.2.1/§6.2 (executed, 0 err). **Role = single-label-failure reproduction + cross-cohort convergence, NOT granular weight transfer** (Kenya symptoms pre-aggregated to 6 groups). **Tanzania D'Acremont retained concise (Mal/Typ); SISA + MAP dropped.** Never in core training/calibration/fairness; all cited (also feeds the ≥3-journal-ref bar). See §6.1 overhaul note + memory `kenya-labeaud-external-dataset`. (The old `external_data/EXTERNAL_DATA_GUIDE.md` documents the superseded SISA/MAP/Tanzania acquisition.)
7. ✅ **Deployment positioning (2026-06-15):** the deliverable is an **uncertainty-aware triage / referral assistant, NOT a detector/diagnostic.** Moderate dengue recall is a *referral trigger*, not a failure. Re-scope is consistent with the problem statement (always triage/refer/recommend-test). §5 conformal+LOSO is the decisive section that earns this framing. See §10 "§5a positioning" + memory `deployment-positioning-triage`.

**Status (2026-06-15):** Phases 0, 1, 1+ (deep-EDA), and **2 (Core Modeling, RQ1+RQ2) COMPLETE** — `notebook.ipynb` §1–§4 written + executed (111 cells, 0 err). §4 results in §10. **Immediate next step:** **§5 Evaluation** (= roadmap Phase 3 + RQ5) — per-label conformal (Malaria/Dengue) + cost-sensitive abstention/referral; **site-aware LOSO** (the honest cross-site numbers) + ROC overlay; full missingness-as-signal CV; fairness audit; V6 phenotype map. This is now the **critical, non-optional** section per the triage positioning.

---

## 20. Glossary

VBD — vector-borne disease · AUFI — acute undifferentiated febrile illness · **Multi-label** — a patient may carry several disease labels at once · **Classifier chains / ECC** — multi-label method modeling label dependencies (ensemble over random orders) · **MNAR / missingness-as-signal** — the *pattern* of which labs are missing may be informative (a test ordered *because* a disease is suspected — watch for leakage) · **Per-label conformal** — produces per-disease prediction with finite-sample coverage (we do NOT claim joint per-patient set coverage) · **Cost-sensitive abstention** — model declines/refers when a confident wrong answer would be dangerous · **Value-of-information / active testing** — choosing the next test that maximizes diagnostic gain per cost · **CDSS / CHW / IMCI / LMIC** — clinical decision-support system / community health worker / WHO Integrated Management of Childhood Illness / low- and middle-income country.

---

*End of PLAN.md — keep updated as phases complete.*
