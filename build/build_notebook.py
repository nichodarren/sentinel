# -*- coding: utf-8 -*-
"""
Builder for notebook.ipynb (CRISP-DM structured, FIT 2026 / SENTINEL).
Phase 1 of writing: SECTION 1 — BUSINESS UNDERSTANDING (subsections 1.1-1.8 + optional 1.9).
Phases 2-6 are scaffolded as headers/placeholders only (content integrated later, per-section).

Build policy (project convention): edit THIS builder, then rebuild the .ipynb. Never hand-edit the .ipynb.
Writing style requirements (from user): formal-academic, flowing narrative, NO em dash, citations on key claims.
"""
import os
import nbformat as nbf

# Project root = parent of this build/ directory (works on any clone, no absolute paths).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ============================================================================
# NOTEBOOK TITLE + CRISP-DM ORIENTATION
# ============================================================================
md(r"""# SENTINEL: A Co-Infection-Aware, Laboratory-Free Triage Engine for Vector-Borne Febrile Disease

### FIT Competition 2026 · Track IV (Human Health / Digital Health Intelligence)
*Symptom-based, co-infection-aware Engine for Non-laboratory Triage in Emergency Low-resource settings*

---

This notebook is organised according to **CRISP-DM** (the Cross-Industry Standard Process for Data Mining; Chapman et al., 2000), the most widely adopted process model for applied data-mining and machine-learning projects. CRISP-DM decomposes a project into six phases that proceed from problem framing to deployment, with deliberate iteration between them. Where the task is specifically a *machine-learning application* rather than classical knowledge discovery, we additionally borrow from **CRISP-ML(Q)** (Studer et al., 2021), an ML-oriented refinement that merges business and data understanding into a single feasibility-driven phase and formalises success criteria on three levels (business, machine-learning, and economic).

| CRISP-DM phase | Where it lives in this notebook | Mapped to our project roadmap |
|---|---|---|
| **1. Business Understanding** | Section 1 (this section) | Problem framing, research questions RQ1-RQ5, success criteria |
| **2. Data Understanding** | Section 2 | Data-quality audit on the raw cohort, leakage and distribution-shift discovery |
| **3. Data Preparation** | Section 3 | Leak-free, evidence-driven preprocessing and feature construction |
| **4. Modeling** | Section 4 | Ensemble Classifier Chains; symptoms-only vs. mask vs. labs (RQ1, RQ2) |
| **5. Evaluation** | Section 5 | Per-label conformal safety, fairness audit, site-aware validation (RQ3, RQ5) |
| **6. Deployment** | Section 6 | Prototype triage interface and an honest generalisation roadmap |

The phase structure is canonical CRISP-DM; the *content* of each phase follows the project roadmap set out in Section 1.8. The reader can therefore navigate the notebook either as a standard data-mining workflow or as a direct answer to the five research questions.
""")

# ============================================================================
# SECTION 1 — BUSINESS UNDERSTANDING
# ============================================================================
md(r"""---

# 1. Business Understanding

> **Purpose of this phase (CRISP-DM).** The first task of the analyst is to understand, from the perspective of the people the project must ultimately serve, what genuinely needs to be accomplished, and to surface early the factors that will decide whether the project succeeds (Chapman et al., 2000). The cost of neglecting this phase is well documented: one expends considerable effort "producing the right answers to the wrong questions." Business Understanding is also where measurable success criteria are fixed, because a criterion that cannot be measured cannot later be evaluated (Studer et al., 2021, invoking the IEEE software-lifecycle principle that requirements be *measurable*).

This section establishes the problem the model must solve, why it matters, and how we will know whether we have solved it. It does so before any data is touched, so that every later modelling decision can be traced back to an objective stated here. The subsections follow the canonical CRISP-DM outputs for this phase, namely *Background*; *Business Objectives and Success Criteria*; *Data-Mining (Machine-Learning) Goals and Success Criteria*; *Inventory of Resources, Assumptions and Constraints*; *Risks and Contingencies*; an *Initial Assessment of Tools, Techniques and Related Work*; and a *Project Plan*. We extend the single success criterion of classical CRISP-DM into the three-level scheme of CRISP-ML(Q), because a clinical-triage tool must be judged not only by a machine-learning metric but by its consequences for patients and for a resource-constrained health system.
""")

# --- 1.1 Background & Problem Context -----------------------------------------
md(r"""## 1.1 Background and Problem Context

The starting point of this project is a recurring and under-served pattern in global health. When a disaster strikes, whether a flood, a cyclone, or mass displacement, its most visible toll is immediate, but a slower and larger burden follows in the weeks afterwards. Stagnant water, crowding, and the breakdown of sanitation create ideal conditions for vector-borne and water-borne febrile disease, so that the disaster is followed by a secondary surge of co-circulating infections such as malaria, dengue, typhoid, and yellow fever (World Health Organization, 2006). The pattern is not hypothetical. Brazil recorded on the order of fifty thousand dengue cases in the aftermath of its 2008 floods, and the 2022 floods in Pakistan were followed by documented epidemics of malaria, dengue, typhoid, and cholera, both consistent with the established link between flooding and post-disaster outbreaks of these diseases (World Health Organization, 2006). The danger is compounded because these infections can co-occur: malaria and dengue co-infection is associated with a roughly fourfold increase in the odds of severe disease in African cohorts (Kotepui et al., 2023). This sequence, in which the long-term health damage of a disaster is mediated by infectious disease, is precisely the concern named in the Track IV primary focus on mitigating "the long-term secondary impacts of disasters on human well-being."

These four diseases share a clinical signature that makes them genuinely hard to tell apart. All present as acute undifferentiated febrile illness, so a clinician confronted with a feverish patient cannot reliably name the cause from symptoms alone, and the diseases can co-occur in a single patient, which a diagnosis that returns only one answer cannot represent. Confirmatory laboratory tests would resolve the ambiguity, but in low-resource and disaster-affected settings these tests are scarce, delayed, or unaffordable, and their scarcity is worst exactly when a surge drives demand to its peak. The default clinical response, to treat every fever as malaria, wastes scarce antimalarials on patients who do not have malaria, delays correct treatment for those who have something else, and contributes to antimicrobial resistance.

Machine learning has been proposed as a way to extend diagnostic reach into these settings, but the prevailing formulation inherits two flaws from the clinical status quo. First, existing models force a single-disease label per patient, which structurally cannot express co-infection. Second, they tend to depend on the very laboratory features that are missing in the field, so their reported accuracy does not survive contact with a lab-scarce deployment. A documented illustration is a Kenyan study of six algorithms in which dengue detection collapsed to between zero and 5.5 percent sensitivity while overall accuracy remained high (Mutai et al., 2023). That failure should be read carefully rather than as a simple indictment of any one design choice: it reflects the single-label framing together with a low dengue prevalence and an accuracy-oriented training objective, so the multi-label reframing this project pursues is best understood as a promising hypothesis for recovering minority-disease detection rather than as a proven remedy. What the example does establish is the pattern this work is built to confront, namely that such models miss dengue and co-infections and fail silently rather than signalling their own uncertainty. The dataset used in this project, an unbenchmarked cohort of 300 patients from the DO and DAFRA health districts of Burkina Faso (Ouedraogo et al., 2025; Mendeley DOI 10.17632/cf49v47z4c.2, CC BY 4.0), exhibits exactly the structural signature that makes the problem hard: symptoms are nearly complete while laboratory fields are sparse, which is the canonical low-resource situation.
""")

# --- 1.2 Business Objectives & Success Criteria -------------------------------
md(r"""## 1.2 Business Objectives and Business Success Criteria

Translating the situation above into a business objective in the CRISP-DM sense means stating, in the language of the people the tool serves, what a successful outcome looks like. The objective of SENTINEL is to give a frontline health worker, operating without reliable laboratory support during or after a disaster, a triage aid that names the *profile* of co-circulating diseases a febrile patient is likely to have, flags the dangerous cases that warrant referral, and does so honestly when it is uncertain. The emphasis on a profile rather than a single label, and on calibrated uncertainty rather than a confident guess, is what distinguishes the objective from the prior art and aligns it with how the diseases actually behave.

The business success criteria, the explicit measures by which a non-technical stakeholder would judge the project, follow directly. SENTINEL should recover the minority and co-infection cases that single-label models miss, most pointedly the patients who have dengue without malaria, since those are the cases the prevailing "treat as malaria" default endangers. It should retain useful accuracy when laboratory features are withheld, since that is the operating condition it is designed for. It should refer dangerous cases conservatively, preferring an explicit "refer for confirmation" over a silent miss. And it should not degrade unfairly across the vulnerable subgroups present in the cohort, in particular children, who dominate the data, and the two health districts, which differ in disease prevalence. These criteria are intentionally qualitative here; their measurable machine-learning counterparts are fixed in Section 1.3, and the alignment between the two levels is what prevents the project from optimising a metric that does not serve the objective.

It is worth recording, in the spirit of the CRISP-DM guidance to note objectives considered and rejected, that we deliberately do not adopt a single-disease accuracy target as a business objective. A high single-label accuracy is achievable on this cohort precisely because malaria is present in ninety percent of patients, and optimising for it would reproduce the failure this project exists to correct. Honest performance on the hard minority cases is the objective, not aggregate accuracy.
""")

# --- 1.3 ML Goals & Success Criteria ------------------------------------------
md(r"""## 1.3 From Business Objective to Machine-Learning Goal

A central task of Business Understanding is to translate the business objective into a data-mining goal, that is, into a problem a model can be trained and measured on (Chapman et al., 2000). CRISP-ML(Q) sharpens this into the requirement that the machine-learning goal carry its own measurable success criterion, defined in alignment with the business criterion so the two cannot pull in opposite directions (Studer et al., 2021).

The reframing at the heart of this project is to treat diagnosis not as single-label classification but as **multi-label prediction of a co-occurring disease profile** over the four diseases, learned from clinical symptoms and from the *pattern* of laboratory missingness rather than from laboratory values that will be absent in deployment. This reframing decomposes into five research questions, the first three of which form the committed backbone of the submission and the last two of which are extensions reserved for the final round. Each question carries a primary success metric chosen to be measurable and to resist the inflation that aggregate accuracy invites on this imbalanced cohort.

| RQ | Machine-learning question | Primary success metric |
|---|---|---|
| **RQ1, Co-infection** | Can co-infection-aware multi-label learning recover minority-disease and co-infection detection where single-label learning fails? | Recall on **non-malarial dengue (n = 14)**, the clinically decisive cell, reported against a single-label baseline with honest confidence intervals. |
| **RQ2, Laboratory-free** | How much predictive signal is available from symptoms alone, and does the laboratory-missingness pattern, treated as a feature, add to it once leakage is controlled? | Symptoms-only performance as the honest ceiling, compared against symptoms-plus-missingness-mask and symptoms-plus-labs, with confirmatory-test masks ablated. |
| **RQ3, Safe triage** | Can per-label calibrated and conformal uncertainty, combined with cost-sensitive abstention, support safe referral of dangerous cases? | Per-disease coverage for the well-powered labels (malaria, dengue) and a dangerous-miss versus referral-rate trade-off; rarer labels reported descriptively. |
| **RQ4, Active testing** *(extension)* | Which single confirmatory test would yield the most diagnostic information per unit cost? | Population-level expected information gain, reported qualitatively and without a per-patient performance claim. |
| **RQ5, Equity and transfer** *(extension)* | Is performance equitable across adequately powered subgroups, and what is an honest path to generalisation beyond a single region? | Child-versus-adult, sex, and site gaps for malaria and dengue, restricted to subgroups of at least twenty patients, with bootstrap confidence intervals. |

The machine-learning success criterion for the submission, the equivalent of a minimum viable product in CRISP-ML(Q) terms, is that the multi-label formulation demonstrably recovers non-malarial dengue recall that a single-label baseline forfeits, while symptoms-only performance is characterised honestly rather than inflated by leakage. A model that merely posts a high aggregate score without moving the hard cell would meet a conventional accuracy target while failing the business objective, which is exactly the outcome the three-level success scheme is designed to catch.
""")

# --- 1.4 Economic / Impact Success Criteria -----------------------------------
md(r"""## 1.4 Economic and Humanitarian Success Criteria

CRISP-ML(Q) recommends attaching an economic success criterion, a key performance indicator that expresses the real-world value of the application rather than its statistical quality (Studer et al., 2021). For a commercial system this is typically a financial return; for a humanitarian system it is more faithfully expressed as impact per unit of a scarce resource. The scarce resource here is the confirmatory laboratory test, and the harm to be reduced is the missed dangerous case.

On those terms the relevant indicators are the number of dangerous cases, above all non-malarial dengue, that the tool surfaces for referral that the malaria-default would have missed, and the laboratory effort that a symptoms-first triage can avoid or target by indicating when confirmation is genuinely warranted rather than testing indiscriminately. These indicators are deliberately framed as decision support at the point of care and at the population level, not as a clinical guarantee, because the cohort is too small and too geographically narrow to support a deployment-grade economic claim. Stating the impact criterion explicitly, even in this qualified form, keeps the project anchored to its humanitarian purpose and prevents a drift toward optimising machine-learning metrics that look impressive but change nothing at the bedside.
""")

# --- 1.5 Situation Assessment: Resources, Assumptions, Constraints ------------
md(r"""## 1.5 Situation Assessment: Resources, Assumptions, and Constraints

The CRISP-DM situation assessment takes inventory of what the project has to work with and the conditions under which it operates, so that later expectations are calibrated to reality rather than to ambition. The principal resource is the dataset itself, a single tabular cohort of 300 patients described by 109 attributes, collected in two health districts of Burkina Faso, recorded in French, and released under an open licence without any prior machine-learning benchmark. Supporting resources are a standard scientific Python environment with the usual modelling and analysis libraries, and a small team operating under a fixed competition deadline. The project is, in resource terms, modest and self-contained, which is itself a constraint on the methods that can be justified.

Several assumptions about the data follow from its provenance and shape, and naming them now prevents them from being mistaken later for findings. We assume that the symptom fields, which are nearly complete, carry most of the learnable signal, and that the laboratory fields, which are sparse, are missing in ways that are themselves informative rather than purely random. We assume that the four target diseases, and not the various catch-all and zero-incidence categories also present, are the proper prediction targets. And we assume that any value recorded in a confirmatory-test field reflects information that would not be available at symptom-based triage time, so such fields must be quarantined from the laboratory-free analysis rather than silently used.

The binding constraints are equally important. The cohort is small, single-region, heavily skewed toward children, and severely imbalanced across diseases, with malaria present in ninety percent of patients, yellow fever in only twelve, and two of the nominal disease categories absent altogether. These properties rule out approaches that depend on abundant or balanced data, including aggressive synthetic oversampling and heavy model stacking or hyperparameter search, and they place a hard ceiling on the strength of any claim the project can responsibly make. The honest consequence, carried consistently through the rest of the notebook, is that SENTINEL is a proof of concept, a methodological framework, and a generalisation roadmap, and not a deployable clinical diagnostic.
""")

# --- 1.6 Risks, Contingencies & Ethical Requirements --------------------------
md(r"""## 1.6 Risks, Contingencies, and Ethical Requirements

CRISP-DM asks the analyst to identify in advance the problems that could derail the project, their consequences, and the actions that would contain them. CRISP-ML(Q) adds that for any application touching people, the non-functional requirements of fairness, legality, and trust belong in this first phase rather than as an afterthought (Studer et al., 2021). Both concerns are acute for a clinical-triage tool built on a small cohort.

The foremost technical risk is data leakage, the contamination of the model by information that encodes the answer or that would be unavailable at prediction time. The cohort contains fields that mirror the diagnosis directly and confirmatory-test results that all but determine the label, and admitting any of these as ordinary features would produce an impressive but meaningless result. The contingency is a disciplined separation: such fields are identified during data understanding, excluded or explicitly quarantined, and the laboratory-free claim is evaluated only on features that would genuinely be present at triage. A second risk is instability of the rare classes, since metrics computed on twelve yellow-fever patients or twenty-nine typhoid patients cannot be trusted; the contingency is to report those labels descriptively, to anchor the headline claim on the better-powered non-malarial dengue cell, and to attach confidence intervals throughout. A third risk, surfaced by the data itself, is distribution shift between the two health districts, which threatens any performance figure pooled naively across them; the contingency is site-aware validation that reports leave-one-site-out performance alongside the pooled figure, so that the gap is exposed rather than hidden.

The ethical requirements are non-negotiable and shape the design rather than merely annotating it. Because the tool concerns human health, it must be audited for equitable performance across the vulnerable subgroups in the data, it must express uncertainty so that a dangerous case is referred rather than silently misclassified, and it must be presented honestly as a non-clinical prototype carrying an explicit disclaimer that it is not for clinical use. These requirements are not a coda to the modelling; they are the reason for the conformal safety layer and the fairness audit that appear in the evaluation phase, and they are what the project means by responsible machine learning in a health setting.
""")

# --- 1.7 Feasibility & Related Work -------------------------------------------
md(r"""## 1.7 Feasibility and Related Work

Before committing to a project, CRISP-ML(Q) recommends an explicit feasibility check, which for a novel application typically takes the form of situating the work against comparable efforts and, where the approach is new to a domain, a proof of concept (Studer et al., 2021). The feasibility of SENTINEL rests on three observations. First, the predictive signal is plausibly present: symptom fields in the cohort are nearly complete, and related work on single-disease febrile-illness classification reports usable discrimination from clinical features, so the question is not whether symptoms carry signal but whether a co-infection-aware formulation can extract more of the clinically important part of it. Second, the methodological ingredients are mature and available: multi-label learning through classifier chains, gradient-boosted trees that handle missingness natively, and conformal prediction for calibrated uncertainty are all established techniques with stable open-source implementations, so the project integrates known methods rather than inventing unproven ones. Third, the data is in hand and openly licensed, which removes the most common cause of premature failure, namely the unavailability of an adequate training sample.

The relationship to prior art is also what locates the project's contribution. Existing approaches fall into single-disease binary classifiers, multiclass single-label models that force one answer per patient, co-infection-as-a-combination-class schemes that cannot generalise to unseen combinations, and rule-based clinical decision systems that depend on point-of-care biomarkers which are frequently unavailable. Each of these falls short of the problem as posed here for a reason that the SENTINEL formulation is designed to remove. The novelty is therefore one of reframing and integration rather than of a single new algorithm, and surveying this related work also satisfies the competition requirement to ground the methodology in at least three peer-reviewed references.
""")

# --- 1.8 Project Plan: CRISP-DM roadmap mapping -------------------------------
md(r"""## 1.8 Project Plan: Mapping the Roadmap onto CRISP-DM

The final output of Business Understanding is a project plan that lists the stages to be executed, their dependencies, and the points at which the process is expected to iterate (Chapman et al., 2000). Our plan is the project roadmap, expressed here as an explicit mapping onto the CRISP-DM phases so that the canonical process and the project's own structure are visibly one and the same. The phases proceed in order, but the arrows between Data Understanding, Data Preparation, and Modeling are intended to be traversed more than once, since evidence found while preparing the data routinely sends the analyst back to re-examine it, which is exactly the iteration CRISP-DM anticipates.

| CRISP-DM phase | Project work | Research questions |
|---|---|---|
| 1. Business Understanding | Problem framing, success criteria, this section | Frames RQ1-RQ5 |
| 2. Data Understanding | Data-quality audit on the raw cohort; leakage, site, and missingness discovery | Evidence for RQ2 |
| 3. Data Preparation | Evidence-driven, leak-free cleaning and feature construction | Enables RQ1, RQ2 |
| 4. Modeling | Ensemble Classifier Chains; symptoms-only vs. mask vs. labs; single-label baseline | RQ1, RQ2 |
| 5. Evaluation | Per-label conformal safety; site-aware validation; fairness audit | RQ3, RQ5 |
| 6. Deployment | Prototype triage interface; generalisation roadmap | RQ4, RQ5, deployment |

This section has fixed the objective, the success criteria on three levels, the constraints and risks, and the plan. Every subsequent phase refers back to a commitment made here, and the first of those phases, Data Understanding, begins by confronting the raw cohort with the assumptions just stated.
""")

# --- 1.9 Terminology (optional, concise) --------------------------------------
md(r"""## 1.9 Terminology

A brief glossary is provided so that a reader unfamiliar with the clinical or methodological vocabulary can follow the argument without recourse to external sources, in keeping with the CRISP-DM terminology output.

- **Vector-borne disease (VBD):** an illness transmitted by a living carrier such as a mosquito; here malaria, dengue, and yellow fever, alongside the water-borne enteric infection typhoid, which co-circulates under the same post-disaster conditions.
- **Acute undifferentiated febrile illness:** fever without localising signs that point to a specific cause, the clinical state in which these diseases are indistinguishable on presentation.
- **Multi-label classification:** a formulation in which each patient may be assigned several disease labels at once, as opposed to single-label classification, which assigns exactly one.
- **Ensemble of Classifier Chains (ECC):** a multi-label method that models dependencies between labels by predicting them in sequence and averaging over several random orderings.
- **Missingness-as-signal:** the use of the pattern of which values are absent, rather than the values themselves, as an informative feature.
- **Conformal prediction:** a framework that converts model scores into prediction sets with a calibrated error rate, used here to express per-label uncertainty for safe referral.
- **Leakage:** the inadvertent inclusion of information that encodes the target or that would be unavailable at prediction time, which inflates apparent performance.
""")

# ============================================================================
# SECTION 2 — DATA UNDERSTANDING  (ported DETECT-only cells from the EDA work)
# ============================================================================
md(r"""---

# 2. Data Understanding

> **Purpose of this phase (CRISP-DM).** Data understanding begins with an initial data collection and proceeds with activities to become familiar with the data, to identify data-quality problems, to discover first insights, and to form hypotheses about hidden structure (Chapman et al., 2000). Its four canonical tasks are to collect the initial data, to describe it, to explore it, and to verify its quality.

A principle governs this section and separates it cleanly from Section 3. Data Understanding *observes and diagnoses* the data as it is; it does not alter it. Every problem surfaced here, an impossible vital sign, a column that leaks the label, a laboratory field whose units are not comparable, is detected and characterised but left untouched, and the repair is deferred to Data Preparation. The bridge between the two phases is the issue-to-solution table in Section 2.6, which records, for each defect found, the remedy that Section 3 will apply. This ordering follows the CRISP-DM guidance that the data-quality task produces a list of problems and *possible* solutions, while the act of cleaning belongs to the preparation phase.

The subsections proceed from the most general view of the data to the most specific. Section 2.1 collects the cohort and records its provenance. Section 2.2 describes its surface properties and per-attribute statistics. Section 2.3 examines the prediction target, whose multi-label structure shapes every later decision. Section 2.4 verifies data quality across four distinct hazards. Section 2.5 explores the cohort for the deeper structure that becomes the modelling hypotheses. Section 2.6 summarises the findings and hands them to Section 3.
""")

# --- Setup: helpers reused by the detection cells (visual + accent-folding) ---
md(r"""### Analysis setup

The cells below load the libraries, the colour palette, and two small helpers used throughout this section. The first applies a consistent house style to every figure; the second folds accents and case so that French categorical labels such as the test result *Negatif* can be matched reliably regardless of trailing spaces or capitalisation. These helpers only support analysis and display; they do not modify the cohort.
""")
code(r'''import warnings, re, unicodedata
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as ipd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)

# Colour palette and per-disease display names used by every figure in this section.
#
# Two separate palettes are kept deliberately distinct so that colour carries one
# meaning at a time. DCOL (defined alongside the target figures) holds the four
# canonical disease colours and is reserved for diseases only. The role palette below
# is used for everything that is NOT a disease (highlights, reference lines, the two
# sites, artefact flags, generic categories). Its hues are chosen to sit outside the
# disease set (red / blue / amber / green) so a slate highlight is never mistaken for
# dengue, nor a danger accent for malaria.
PRIMARY, SECONDARY = "#3D5A80", "#8A94A6"      # slate highlight, neutral grey
POSITIVE, WARNING, NEGATIVE = "#2E8B57", "#B0833B", "#9B2D6F"   # good, caution-brown, danger-mulberry
SITE_A, SITE_B = "#5E548E", "#C77DA0"          # the two sites (purple pair, non-disease)
DISP = {"malaria": "Malaria", "dengue": "Dengue", "yellow_fever": "Yellow fever", "typhoid": "Typhoid"}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "figure.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": "#222", "axes.edgecolor": "#BBB",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#444", "ytick.color": "#444", "legend.frameon": False, "legend.fontsize": 8.5,
})

def style_ax(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    """Apply the house style to an axis (grid on one axis only). grid_axis in {x,y,both,none}."""
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(axis=grid_axis, alpha=0.9)
        if grid_axis in ("x", "y"):
            ax.grid(axis=("x" if grid_axis == "y" else "y"), visible=False)
    ax.tick_params(length=0)
    return ax

def bar_labels(ax, bars, fmt="{:.0f}", dx=0, dy=3, fs=8.5, color="#222"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x()+b.get_width()/2, h), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs, color=color)

def fold_raw(s):
    """Accent-fold, upper-case and strip a raw label so French values match reliably. Display only."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).strip().upper()

# A small number of figures in this notebook are interactive. They are rendered with
# the plotting library's JavaScript embedded directly in the notebook, so they stay
# interactive when the notebook is run yet never blank on a static or offline view,
# and each one also writes a still image to disk so the report can reproduce it.
import os as _os
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook_connected"
_os.makedirs("eda_figs", exist_ok=True)
PLOTLY_FONT = dict(family="DejaVu Sans, Arial, sans-serif", size=12, color="#222")

def show_interactive(fig, name, height=460, width=940, title=None):
    """Render a Plotly figure inline (embedded JS) with the house margins and a centred
    title, and export a still PNG fallback to eda_figs/ for the report."""
    fig.update_layout(template="plotly_white", font=PLOTLY_FONT,
                      margin=dict(l=70, r=40, t=72, b=60), height=height, width=width,
                      title=dict(text=title or "", x=0.5, xanchor="center",
                                 font=dict(size=15, color="#1b1b1b")))
    try:
        fig.write_image(f"eda_figs/{name}.png", scale=2)
    except Exception as _e:
        print(f"[note] static export skipped for {name}: {_e}")
    from plotly.io import to_html
    from IPython.display import HTML, display
    display(HTML(to_html(fig, include_plotlyjs="inline", full_html=False)))

print("Analysis setup loaded: palette, style_ax(), bar_labels(), fold_raw(), show_interactive().")
''')

# --- 2.1 Initial Data Collection ---------------------------------------------
md(r"""## 2.1 Initial Data Collection

The cohort is a single tabular file of 300 patients described by 109 attributes, collected in the DO and DAFRA health districts of the Hauts-Bassins region of Burkina Faso and released as open data (Ouedraogo et al., 2025). The file is semicolon-separated and uses the comma as its decimal mark, which is the French regional convention, and it is encoded in UTF-8.

The encoding deserves an explicit note, because it is a trap that silently corrupts the analysis if handled carelessly. The accented bytes in the file are genuine UTF-8, so a accented test result such as *Negatif* is stored correctly. Reading the file under a Western European single-byte encoding leaves the column headers looking plausible while turning every accented data cell into mojibake, which in turn causes the accented categorical values to fail every later match and collapse silently to missing. The load below therefore fixes the encoding explicitly and asserts the expected shape, so that any future change to the source file is caught immediately rather than discovered as a constant feature much later.
""")
code(r'''RAW_PATH = "data.csv"
LOAD_KW = dict(sep=";", decimal=",", encoding="utf-8")

raw = pd.read_csv(RAW_PATH, **LOAD_KW)
raw.columns = [c.strip() for c in raw.columns]
print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
assert raw.shape == (300, 109), "Unexpected raw shape - re-check the source file."
print("Load parameters verified (utf-8, sep=';', decimal=',').")
''')

md(r"""The 109 columns are referenced by position throughout this section, so a guard verifies that a set of known columns still sit where the analysis expects them. The same cell records the roles that the rest of the section relies on: the two identifier columns, the concatenated-diagnosis column, the eight disease one-hot columns, and the four that form the prediction target. Three of the nominal diseases, chikungunya, Zika, and the eighth unlabelled option, have no positive cases at all and are recorded here only so that their absence is explicit rather than assumed.
""")
code(r'''C = list(raw.columns)   # C[i] = verified name of column i

_EXPECTED_AT = {
    1: "Genre", 2: "ge (Age)", 3: "Poids", 4: "Mua", 8: "fièvre", 63: "Température axillaire",
    64: "respiratoire", 65: "pouls", 66: "artérielle", 67: "capillaire", 68: "TDR",
    69: "Goutte", 73: "Hematocrite", 77: "Lymphocytes", 79: "globules blancs",
    80: "plaquettaire", 81: "Neutrophiles", 83: "Créatinine", 96: "Household", 97: "Dengua",
    98: "Maladies diagnostiquées", 99: "Paludisme", 100: "/Dengue", 101: "Chikun",
    102: "jaune", 103: "Typho", 104: "Zika", 105: "Autres", 106: "Option 8", 108: "uuid",
}
_bad = {i: C[i] for i, frag in _EXPECTED_AT.items() if frag not in C[i]}
assert not _bad, f"Column schema changed - positional bindings invalid at: {_bad}"
print(f"Column-name guard passed ({len(_EXPECTED_AT)} positions verified).")

ID_COLS         = [C[0], C[108]]   # 'Centre de sante', '_uuid'
COMBINED_DX_COL = C[98]            # concatenated diagnosis labels
SITE_COL        = C[0]            # 'Centre de sante' (health district)

DISEASE_COLS = {
    "malaria": C[99], "dengue": C[100], "chikungunya": C[101], "yellow_fever": C[102],
    "typhoid": C[103], "zika": C[104], "others": C[105], "option_8": C[106],
}
TARGET_LABELS = ["malaria", "dengue", "yellow_fever", "typhoid"]
DROP_DISEASES = ["chikungunya", "zika", "option_8"]   # all-zero

LEAK_COL_DENGUE  = C[97]            # 'Dengue (Dengua)' (suspected label copy)
HOUSEHOLD_DENGUE = C[96]            # 'Household Dengue' (genuine history)
CONFIRMATORY_TESTS = [C[68], C[69]] # 'Test TDR', 'Goutte epaisse'

print("Disease block positive counts (raw):")
for name, col in DISEASE_COLS.items():
    pos = int((raw[col] == 1).sum())
    tag = "  <- drop (all-zero)" if name in DROP_DISEASES else ("  <- target" if name in TARGET_LABELS else "")
    print(f"  {name:13s}: {pos:3d} positives{tag}")
print(f"\nRows with ALL disease columns NaN (missing label): "
      f"{raw[list(DISEASE_COLS.values())].isna().all(axis=1).sum()}")
''')

# --- 2.2 Data Description ------------------------------------------------------
md(r"""## 2.2 Data Description

Before any quality judgement, the data must be described at the surface level, which the CRISP-DM describe-data task defines as the format, the quantity, the identity of the fields, and the basic per-attribute statistics. This subsection answers three questions in turn: how large the dataset is and what kinds of variables it contains, what the records actually look like, and how each variable is distributed. The intent is orientation, not yet diagnosis; the defects these descriptions begin to hint at are confirmed in Section 2.4.

The cohort divides naturally into a small number of variable families. The large majority of the columns are binary symptom indicators recorded as the French *OUI* or *NON*, a smaller group holds numeric vitals and laboratory measurements, and the remainder are categorical fields such as the test results and the patient identifiers. The inventory below assigns each column to one of these families by inspecting its values, which both quantifies the composition of the dataset and exposes the first structural fact of the problem: the symptom block is wide and almost complete, while the laboratory block is narrow and sparse.
""")
code(r'''def classify_column(s):
    """Assign a column to a coarse family by inspecting its non-null values (observation only)."""
    nun = s.dropna().map(lambda v: fold_raw(v))
    vals = set(nun.unique())
    if vals and vals <= {"OUI", "NON"}:
        return "binary symptom (OUI/NON)"
    num = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if num.notna().mean() >= 0.80:
        return "numeric (vital / lab)"
    if {"POSITIF", "NEGATIF"} & vals:
        return "test result (Positif/Negatif)"
    return "categorical / text"

families = pd.Series({c: classify_column(raw[c]) for c in raw.columns})
inv = (families.value_counts()
       .rename_axis("variable family").reset_index(name="n columns"))
print(f"Dataset dimension: {raw.shape[0]} patients x {raw.shape[1]} attributes\n")
ipd.display(inv)

miss = raw.isna().mean()
print(f"Columns with under 5% missing values: {(miss < 0.05).sum()} of {raw.shape[1]} "
      f"({(miss < 0.05).mean():.0%})")
print("Binary symptom columns are near-complete; the sparsest fields are laboratory and "
      "anthropometric measurements (detailed in Section 2.4).")
''')

md(r"""The column inventory below names every field, its inferred family, its non-null count, and its number of distinct values. It is the catalogue that the positional references in this section ultimately point to, and it makes the meaning of each attribute explicit in one place rather than leaving it implicit in the code.
""")
code(r'''catalogue = pd.DataFrame({
    "family": families,
    "non_null": raw.notna().sum(),
    "n_unique": raw.nunique(dropna=True),
    "% missing": (raw.isna().mean() * 100).round(1),
}).reset_index(names="column")
ipd.display(catalogue)
''')

md(r"""A direct look at a few records grounds the description in the actual data. The preview below shows a compact selection of columns spanning the families above, so that the form of a symptom indicator, a vital sign, and a disease label can be seen side by side.
""")
code(r'''preview_cols = [C[1], C[2], C[3], C[8], C[63], C[65], C[80], C[97], C[99], C[100], C[103]]
ipd.display(raw[preview_cols].head(8))
''')

md(r"""The numeric description that follows reports, for each quantitative field, the standard summary statistics together with skewness and kurtosis. The vitals and anthropometry are parsed from their raw strings for this purpose only. Two patterns are worth noting in advance of the formal quality check. Several fields show maxima that are physically impossible for a human, which is the signature of data-entry corruption examined in Section 2.4.1, and several laboratory fields span several orders of magnitude, which foreshadows the unit-comparability problem of Section 2.4.3.
""")
code(r'''def to_num(col):
    return pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False)
                         .str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

numeric_fields = {
    "age": C[2], "weight (kg)": C[3], "MUA (cm)": C[4], "temperature (C)": C[63],
    "resp rate": C[64], "pulse (bpm)": C[65], "platelets": C[80], "hematocrit": C[73],
    "wbc": C[79], "neutrophils": C[81], "lymphocytes": C[77], "creatinine": C[83],
}
num_df = pd.DataFrame({name: to_num(col) for name, col in numeric_fields.items()})
desc = num_df.describe().T
desc["iqr"] = desc["75%"] - desc["25%"]
desc["skew"] = num_df.skew()
desc["kurtosis"] = num_df.kurtosis()
desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
ipd.display(desc[["count", "mean", "std", "min", "q1", "median", "q3", "max", "iqr", "skew", "kurtosis"]].round(2))
''')

md(r"""For the categorical and binary fields, the analogous description reports the number of non-null values, the number of distinct categories, the most frequent value, and its frequency. For the binary symptom block this confirms that the fields are genuinely two-valued and reveals, through the modal value, that most symptoms are absent in most patients, which is the expected sparsity of a symptom checklist.
""")
code(r'''cat_cols = [c for c in raw.columns if families[c] in
            ("binary symptom (OUI/NON)", "test result (Positif/Negatif)", "categorical / text")]
def mode_info(s):
    vc = s.dropna().value_counts()
    top = vc.index[0] if len(vc) else None
    return pd.Series({"non_null": int(s.notna().sum()), "n_unique": int(s.nunique(dropna=True)),
                      "top (mode)": top, "freq": int(vc.iloc[0]) if len(vc) else 0,
                      "proportion": round(vc.iloc[0] / s.notna().sum(), 3) if len(vc) and s.notna().sum() else np.nan})
cat_desc = pd.DataFrame({c: mode_info(raw[c]) for c in cat_cols}).T
ipd.display(cat_desc.head(30))
print(f"(showing 30 of {len(cat_cols)} categorical/binary columns; the binary symptom block dominates)")
''')

# --- 2.3 Target Understanding -------------------------------------------------
md(r"""## 2.3 Target Understanding

The describe-data task gives particular weight to the distribution of the target attribute, and for this problem the target is not a single column but a profile across four diseases. Understanding its structure is the most consequential observation in the whole section, because the geometry of the labels determines which formulations are viable and which evaluation figure is honest. This subsection therefore reads the four disease one-hot columns directly from the raw data and characterises three properties: how the labels nest within one another, how sparse the joint label space is, and whether any single symptom could separate a disease on its own.

The four target columns are already coded as one and zero, so no cleaning is required to inspect them. One patient carries no disease label at all, a defect noted here and resolved in Section 2.4.4; that row is set aside for the target geometry below so that the counts reflect the patients who can actually supervise a model.
""")
code(r'''# Build the target one-hot directly from raw (observation only; the cohort is not modified here).
Y_raw = pd.DataFrame({l: pd.to_numeric(raw[DISEASE_COLS[l]], errors="coerce") for l in TARGET_LABELS})
labelled = Y_raw.notna().any(axis=1)
Y = Y_raw[labelled].fillna(0).astype(int)   # local, in-memory view for target analysis
print(f"Patients with at least one disease label: {int(labelled.sum())} of {len(raw)} "
      f"(one all-missing row is set aside; see Section 2.4.4)\n")

# Per-disease prevalence and how each minority disease nests inside malaria.
print("Disease prevalence and nestedness within malaria:")
for l in TARGET_LABELS:
    n = int((Y[l] == 1).sum())
    if l == "malaria":
        print(f"  {DISP[l]:13s}: {n:3d} positive ({n/len(Y):.0%})")
    else:
        k = int((Y.loc[Y[l] == 1, "malaria"] == 1).sum())
        print(f"  {DISP[l]:13s}: {n:3d} positive ({n/len(Y):4.0%})  of which {k} co-occur with malaria "
              f"-> non-malarial n={n-k}")
''')

md(r"""The number of labels carried per patient, and the number of distinct disease combinations that actually occur, decide whether the problem can be treated as a single multi-class label over combinations or must be modelled as several related binary labels. The result below is decisive: only a handful of the sixteen possible combinations appear, and they are heavily concentrated on malaria and malaria-plus-dengue, so treating each combination as its own class would fragment the already scarce data and could never generalise to a combination not seen in training.
""")
code(r'''# Labels per patient and joint label-space occupancy.
per_patient = Y.sum(axis=1).value_counts().sort_index()
print("Labels per patient:")
for k, v in per_patient.items():
    print(f"  {k} disease(s): {v:3d} patients")

combo = Y.apply(lambda r: "+".join(DISP[l] for l in TARGET_LABELS if r[l] == 1) or "(none)", axis=1)
vc = combo.value_counts()
probs = (vc / vc.sum()).values
H = -np.sum(probs * np.log2(probs))
print(f"\nJoint label space: {len(vc)} of 16 possible combinations occur; "
      f"entropy {H:.2f} bits; {int((vc <= 3).sum())} combinations occur 3 times or fewer.")
print("Most frequent combinations:")
ipd.display(vc.head(6).rename_axis("disease profile").reset_index(name="patients"))
''')

md(r"""The table states the concentration of the label space; the figure below lets the eye see it. Each column is one disease profile that actually occurs, the matrix beneath marks which diseases make up that profile, and the bar above gives the number of patients who carry it. The picture confirms that only a handful of the sixteen possible combinations appear and that they are dominated by malaria and malaria with dengue, which is the empirical reason for modelling related binary labels rather than treating each combination as its own class.

The profiles drawn in red are the ones that decide the central claim, namely the patients who carry dengue while malaria is absent. That set is not a single bar but the union of every profile in which dengue is present and malaria is absent: the dengue-only profile, the dengue-with-yellow-fever profile, and the dengue-with-typhoid profile. Summed across them it comes to fourteen patients, the smallest cell that can still serve as a fair test of whether the model recovers minority-disease detection where the single-label framing fails, and the dengue-only profile alone holds eleven of those fourteen.
""")
code(r'''# UpSet-style view of the disease-combination space (built from Y; observation only).
labels4 = ["malaria", "dengue", "typhoid", "yellow_fever"]
combo_v1 = Y[labels4].apply(lambda r: tuple(int(r[l]) for l in labels4), axis=1)
counts_v1 = combo_v1.value_counts()
counts_v1 = counts_v1[[k for k in counts_v1.index if sum(k) > 0]]   # drop the empty profile
order_v1 = counts_v1.index.tolist()

def _is_nmd(k):   # dengue present, malaria absent: the honest yardstick set
    d = dict(zip(labels4, k)); return d["dengue"] == 1 and d["malaria"] == 0

fig = plt.figure(figsize=(9, 5.2))
gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.4], hspace=0.06)
axb = fig.add_subplot(gs[0]); axm = fig.add_subplot(gs[1], sharex=axb)
xs = np.arange(len(order_v1))
bars = axb.bar(xs, [counts_v1[k] for k in order_v1],
               color=[NEGATIVE if _is_nmd(k) else PRIMARY for k in order_v1], zorder=3)
bar_labels(axb, bars, fmt="{:.0f}", dy=2, fs=8)
style_ax(axb, ylabel="patients", grid_axis="y"); axb.set_ylim(0, max(counts_v1) * 1.18)
axb.set_title("Only a handful of disease profiles actually occur", loc="center", pad=10, fontsize=15)
plt.setp(axb.get_xticklabels(), visible=False)
for yi, l in enumerate(labels4):
    axm.scatter(xs, [yi] * len(xs), s=26, color="#DDD", zorder=1)
for xi, k in enumerate(order_v1):
    on = [yi for yi, l in enumerate(labels4) if dict(zip(labels4, k))[l] == 1]
    col = NEGATIVE if _is_nmd(k) else "#333"
    axm.scatter([xi] * len(on), on, s=46, color=col, zorder=3)
    if len(on) > 1:
        axm.plot([xi, xi], [min(on), max(on)], color=col, lw=1.6, zorder=2)
axm.set_yticks(range(len(labels4))); axm.set_yticklabels([DISP[l] for l in labels4])
axm.set_xticks(xs); axm.set_xticklabels([])   # counts already labelled on the bars above
axm.tick_params(axis="x", length=0)
axm.set_ylim(-0.6, len(labels4) - 0.4); axm.invert_yaxis()
axm.grid(False); axm.tick_params(length=0)
for sp in axm.spines.values(): sp.set_visible(False)
plt.show()
print(f"Non-malarial dengue (red profiles, dengue present and malaria absent): "
      f"n={int(sum(counts_v1[k] for k in order_v1 if _is_nmd(k)))}")
''')

md(r"""The nestedness is easier to grasp as a picture than as a list of ratios. The figure below splits each minority disease into the part that co-occurs with malaria and the part that does not. The non-malarial portion is the genuinely hard target, and the figure makes plain how small it is: typhoid and yellow fever leave only a handful of non-malarial cases each, too few to learn from symptoms, while non-malarial dengue, at fourteen patients, is the smallest cell that can still serve as an honest test of the central claim.
""")
code(r'''DCOL = {"malaria": "#C1442E", "dengue": "#2E6FC1", "yellow_fever": "#E6A817", "typhoid": "#4B9E5F"}
order = ["dengue", "typhoid", "yellow_fever"]
fig, ax = plt.subplots(figsize=(8, 3.0)); ax.set_axisbelow(True); ys = np.arange(len(order))
for k, l in enumerate(order):
    pos = Y[l] == 1; n = int(pos.sum()); withm = int((Y.loc[pos, "malaria"] == 1).sum()); nonm = n - withm
    ax.barh(k, withm, color=DCOL["malaria"], label="co-occurs with malaria" if k == 0 else None, zorder=3)
    ax.barh(k, nonm, left=withm, color=DCOL[l], label=None, zorder=3)
    ax.annotate(f"non-malarial n={nonm}", (n + 0.6, k), va="center", fontsize=8.5, color="#222",
                fontweight="bold" if l == "dengue" else "normal")
    ax.annotate(f"{withm}", (withm / 2, k), va="center", ha="center", color="white", fontsize=8)
ax.set_yticks(ys); ax.set_yticklabels([DISP[l] for l in order]); ax.set_xlim(0, 78)
style_ax(ax, xlabel="patients", grid_axis="x")
ax.set_title("The minority diseases are nearly contained within malaria", loc="center", pad=12, fontsize=15)
ax.legend(loc="upper right", framealpha=0.9, borderpad=0.5, fontsize=8.5)
plt.show()
''')

md(r"""Finally, a floor for any model is whether a single symptom could separate a disease unaided. Using dengue as the test case, the best single symptom is scored by Youden's J, which combines sensitivity and specificity into one number that is zero when a feature is no better than chance. The best symptom scores close to zero, which means no individual symptom discriminates dengue and a multivariate model is genuinely required rather than a simple rule. The symptom indicators are encoded from their raw form for this calculation only.
""")
code(r'''# Encode the binary symptom block from raw (in-memory, for this ceiling calculation only).
def encode_binary(col):
    return raw[col].map(lambda v: {"OUI": 1.0, "NON": 0.0}.get(fold_raw(v), np.nan) if pd.notna(v) else np.nan)

# Exclude the leakage column flagged in Section 2.4.2: it is a copy of the dengue label,
# not a symptom, so it must not enter the single-symptom ceiling or the information analysis.
SYM = [c for c in raw.columns if families[c] == "binary symptom (OUI/NON)" and c != LEAK_COL_DENGUE]
Xsym = pd.DataFrame({c: encode_binary(c) for c in SYM})[labelled.values].reset_index(drop=True)
Yd = Y["dengue"].reset_index(drop=True)
def youden(s):
    x = Xsym[s].fillna(0).astype(int)
    tp = ((x == 1) & (Yd == 1)).sum(); fn = ((x == 0) & (Yd == 1)).sum()
    tn = ((x == 0) & (Yd == 0)).sum(); fp = ((x == 1) & (Yd == 0)).sum()
    return tp / max(1, tp + fn) + tn / max(1, tn + fp) - 1
js = {s: youden(s) for s in SYM if Xsym[s].sum() >= 10}
best = max(js, key=js.get)
print(f"Best single symptom for dengue: '{best}'  Youden J = {js[best]:.2f}")
print("A Youden J near zero means no single symptom separates dengue; a multivariate model is required.")
''')

md(r"""These observations fix three commitments that the modelling phase will inherit. Typhoid and yellow fever are so nearly contained within malaria that they can only be reported descriptively, the honest measure of success on the minority target is recall on non-malarial dengue, and because the label combinations are few and concentrated the problem must be modelled as related binary labels rather than as a single class over combinations, with malaria conditioned first.
""")

# --- 2.4 Data Quality Verification --------------------------------------------
md(r"""## 2.4 Data Quality Verification

The verify-data-quality task asks whether the data is complete, whether it contains errors and how common they are, and how missing values are represented and distributed. This subsection answers those questions across four distinct hazards that a cohort of this kind is prone to: physically impossible numeric entries, a feature that secretly copies the label, laboratory fields whose units are not comparable, and redundant or unlabelled records. In keeping with the principle stated at the head of this section, each hazard is detected and quantified but not repaired. The remedy for each is recorded in the summary table of Section 2.6 and carried out in Section 3.
""")

md(r"""### 2.4.1 Physically impossible numeric entries

The first hazard is data-entry corruption in the numeric fields. Each vital and anthropometric field is parsed numerically and screened against wide physiological bounds, chosen deliberately loose so that a genuine febrile extreme is never mistaken for an error. The values that fall outside these bounds are not borderline cases; they are temperatures recorded with a dropped decimal point, a blood-pressure reading run together into a single number, and an impossible pulse. The screen below lists how many impossible values each field contains and shows examples of the raw strings, which reveal the mechanical nature of the corruption.
""")
code(r'''haz_specs = [
    ("temperature (C)", C[63], 34, 43), ("pulse (bpm)", C[65], 30, 220),
    ("resp rate", C[64], 8, 80), ("capillary refill (s)", C[67], 0, 10),
    ("MUA (cm)", C[4], 5, 40), ("weight (kg)", C[3], 1, 200),
]
rows = []
for name, col, lo, hi in haz_specs:
    s = pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^0-9.]", "", regex=True),
                      errors="coerce")
    bad = s[(s < lo) | (s > hi)].dropna()
    examples = ", ".join(map(lambda x: str(int(x)) if x == int(x) else str(x), bad.head(4).tolist())) or "(none)"
    rows.append((name, f"[{lo},{hi}]", int(bad.shape[0]), examples))
ipd.display(pd.DataFrame(rows, columns=["field", "physiologic bound", "# impossible", "example raw values"]))

bp_bad = raw[C[66]].astype(str)
print("Blood-pressure raw hazards:",
      [v for v in bp_bad.unique() if any(k in v for k in ["10976", "|"]) or (v.replace(".", "").isdigit() and len(v) >= 5)][:6])
print("Temperature dropped-decimal hazards:",
      [v for v in raw[C[63]].astype(str).unique() if v.replace(".", "").isdigit() and float(v.replace(",", ".")) > 100][:6])
print("\nThese are mechanical typos, not extreme patients; Section 3 repairs each value from itself.")
''')

md(r"""### 2.4.2 A feature that copies the label

The second hazard is target leakage, where a column that looks like an ordinary feature in fact encodes the answer. The column named *Dengue (Dengua)* is positioned among the medical-history fields and might be taken for a history of dengue, but its name is suspicious. The test is to cross-tabulate it against the dengue label: if every positive value of the column corresponds to a positive dengue label, the column is a copy of the target rather than a clinical history, and training on it would inflate every metric while collapsing on unseen data. The crosstab and the figure below show exactly that pattern, an empty off-diagonal cell that betrays a perfect reconstruction of the label.
""")
code(r'''dengua = raw[C[97]].map(lambda v: fold_raw(v) if pd.notna(v) else np.nan)
dengue_lbl = pd.to_numeric(raw[C[100]], errors="coerce")
ct = pd.crosstab(dengua, dengue_lbl, dropna=False)
ipd.display(ct)
oui_pos = int(((dengua == "OUI") & (dengue_lbl == 1)).sum()); oui_tot = int((dengua == "OUI").sum())
print(f"'Dengue (Dengua)' = OUI -> {oui_pos}/{oui_tot} are dengue-positive ({oui_pos/max(1,oui_tot):.0%}); "
      f"captures {oui_pos}/{int((dengue_lbl==1).sum())} of all dengue cases.")
print("Also flagged: the concatenated-diagnosis column and the free-text 'other illnesses' field, which name the diagnosis.")
''')
code(r'''ctv = pd.crosstab(dengua.fillna("NaN"), dengue_lbl).reindex(index=["OUI", "NON"], columns=[1, 0])
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.imshow(ctv.values, cmap="Reds", aspect="auto")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Dengue = 1", "Dengue = 0"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Dengua = OUI", "Dengua = NON"])
for r in range(2):
    for c in range(2):
        val = int(ctv.values[r, c])
        ax.text(c, r, val, ha="center", va="center", fontsize=14,
                color="white" if val > ctv.values.max() * 0.5 else "#222",
                fontweight="bold" if (r, c) == (0, 0) else "normal")
style_ax(ax, grid_axis="none")
ax.set_title("'Dengua' is a copy of the label", loc="center", pad=12, fontsize=15)
plt.show()
print("The empty OUI and Dengue=0 cell means zero false positives: this column reconstructs the label (a hard leak).")
''')

md(r"""### 2.4.3 Laboratory fields with non-comparable units

The third hazard concerns the quantitative laboratory fields, which may pool several analytes or units within a single column so that their absolute values are not comparable across patients. The test is to plot each field on a logarithmic axis and look for the difference between a clean separation into a small number of unit clusters, which can be reconciled, and a continuous smear across several orders of magnitude, which cannot. Only the platelet count shows a clean bimodal split with an empty gap, consistent with two recording units that differ by a factor of a thousand. The remaining labs spread across four to five decades, which is the signature of genuinely non-comparable measurements that no rescaling can fix.
""")
code(r'''labcols = {"platelets": C[80], "hematocrit": C[73], "wbc": C[79],
           "neutrophils": C[81], "lymphocytes": C[77], "creatinine": C[83]}
fig, axes = plt.subplots(2, 3, figsize=(12, 5.6))
spans = {}
for ax, (nm, col) in zip(axes.ravel(), labcols.items()):
    v = pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False), errors="coerce").dropna()
    v = v[v > 0]
    spans[nm] = (v.min(), v.max(), np.log10(v.max()) - np.log10(v.min()))
    sv = np.sort(v.values); k = int(np.argmax(sv[1:] / sv[:-1])); n_above = len(sv) - k - 1
    balanced = min(k + 1, n_above) / len(sv) >= 0.20
    col_c = POSITIVE if nm == "platelets" else WARNING
    ax.hist(np.log10(v), bins=25, color=col_c, alpha=0.85, zorder=3)
    if nm in ("platelets", "creatinine"):
        ax.axvspan(np.log10(sv[k]), np.log10(sv[k + 1]), color=PRIMARY, alpha=0.12, zorder=2)
        gap_mid = np.log10(np.sqrt(sv[k] * sv[k + 1])); ytop = ax.get_ylim()[1]
        col_a = PRIMARY if balanced else "#777"
        note = ("two unit clusters\n(x1000 per uL vs per uL)\n-> x1000 fix" if nm == "platelets"
                else "1 outlier,\nnot a cluster\n-> no safe rule")
        ax.annotate(f"{n_above}/{len(sv)}\nright of gap", (gap_mid, ytop * 0.92), ha="center", va="top",
                    fontsize=8, color=col_a, fontweight=("bold" if balanced else "normal"))
        ax.annotate(note, (gap_mid, ytop * 0.55), ha="center", va="top", fontsize=7, color=col_a)
    style_ax(ax, xlabel="log10 value", grid_axis="y"); ax.set_axisbelow(True); ax.set_ylabel("")
    ax.set_title(nm, loc="center", fontsize=13)
fig.suptitle("Only platelets split into two clean unit clusters", y=0.99, fontsize=15, fontweight="bold", x=0.5, ha="center")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
plat = pd.to_numeric(raw[C[80]].astype(str).str.replace(",", ".", regex=False), errors="coerce").dropna()
gap_lo, gap_hi = plat[plat < 1000].max(), plat[plat >= 1000].min()
print(f"platelets: clean bimodal split, low cluster max={gap_lo:.0f}, high cluster min={gap_hi:.0f} "
      f"(empty gap {gap_lo:.0f} to {gap_hi:.0f}); the low cluster is in thousands per uL.")
print("orders of magnitude spanned per lab (max/min):")
for nm, (lo, hi, dec) in spans.items():
    print(f"  {nm:12s}: {lo:.1f} ... {hi:.1f}   (~{dec:.1f} log10 decades)")
''')

md(r"""### 2.4.4 Redundant columns and unlabelled records

The final hazard is redundancy and incompleteness among the discrete fields. Two pairs of columns appear to duplicate one another, the two headache fields and the two frequent-urination fields, and the question is whether each pair is a true duplicate that should be merged or two distinct concepts that should both be kept. Agreement and Cohen's kappa measure how often the pair coincides, and per-disease odds ratios test whether they carry the same signal. The headache pair agrees only moderately and, decisively, the two columns point at different diseases, so they are not duplicates; the urination pair agrees almost perfectly and is a genuine duplicate. The same cell scans the disease block for records with no label at all and finds exactly one.
""")
code(r'''from sklearn.metrics import cohen_kappa_score
def oui(col): return raw[col].map(lambda v: {"OUI": 1.0, "NON": 0.0}.get(fold_raw(v), np.nan) if pd.notna(v) else np.nan)
def quick_or(x, y):
    x = x.fillna(0); a = ((x == 1) & (y == 1)).sum(); b = ((x == 1) & (y == 0)).sum()
    c = ((x == 0) & (y == 1)).sum(); d = ((x == 0) & (y == 0)).sum()
    return (a + .5) * (d + .5) / ((b + .5) * (c + .5))

h1, h2 = oui(C[10]), oui(C[33])
k_h = cohen_kappa_score(h1.fillna(0).astype(int), h2.fillna(0).astype(int))
print(f"HEADACHE  'Maux de tete' (n={int(h1.sum())}) vs 'Cephalee' (n={int(h2.sum())}): "
      f"agreement={(h1.fillna(0) == h2.fillna(0)).mean():.0%}, Cohen kappa={k_h:.2f}")
print("  per-disease odds ratio (do they point differently?):")
for l in TARGET_LABELS:
    y = pd.to_numeric(raw[DISEASE_COLS[l]], errors="coerce").fillna(0)
    print(f"    {l:13s}: OR(Maux)={quick_or(h1, y):5.2f}   OR(Cephalee)={quick_or(h2, y):5.2f}")

u1, u2 = oui(C[35]), oui(C[38])
print(f"\nURINATION  col35 (n={int(u1.sum())}) vs col38 (n={int(u2.sum())}): "
      f"agreement={(u1.fillna(0) == u2.fillna(0)).mean():.0%}, "
      f"Cohen kappa={cohen_kappa_score(u1.fillna(0).astype(int), u2.fillna(0).astype(int)):.2f}")

allnan = raw[list(DISEASE_COLS.values())].isna().all(axis=1)
print(f"\nLABEL QUALITY: rows with ALL disease labels missing: {int(allnan.sum())} -> raw index {list(raw.index[allnan])}")
''')
code(r'''dis_y = {l: pd.to_numeric(raw[DISEASE_COLS[l]], errors="coerce").fillna(0) for l in TARGET_LABELS}
or_maux = [quick_or(h1, dis_y[l]) for l in TARGET_LABELS]
or_ceph = [quick_or(h2, dis_y[l]) for l in TARGET_LABELS]
fig, ax = plt.subplots(figsize=(8, 3.4)); ax.set_axisbelow(True); ys = np.arange(len(TARGET_LABELS)); hh = 0.38
b1 = ax.barh(ys + hh / 2, or_maux, hh, color=PRIMARY, label="'Maux de tete' (kept)", zorder=3)
b2 = ax.barh(ys - hh / 2, or_ceph, hh, color=SECONDARY, label="'Cephalee' (kept)", zorder=3)
ax.axvline(1, color=NEGATIVE, ls="--", lw=1, label="OR = 1 (no association)", zorder=4)
ax.set_yticks(ys); ax.set_yticklabels([DISP[l] for l in TARGET_LABELS])
for bars, vals in [(b1, or_maux), (b2, or_ceph)]:
    for bar, v in zip(bars, vals):
        dx = 9 if 0.8 <= v <= 1.15 else 3
        ax.annotate(f"{v:.1f}", (v, bar.get_y() + bar.get_height() / 2), xytext=(dx, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#222")
style_ax(ax, xlabel="odds ratio vs disease", grid_axis="x")
ax.set_title("The two 'headache' columns carry different signal", loc="center", pad=12, fontsize=15)
ax.legend(loc="lower right")
plt.show()
print("'Maux de tete' is diagnostic (dengue/typhoid odds well above 1) while 'Cephalee' sits at OR near 1: not duplicates.")
''')

# --- 2.5 Exploratory Insight & Hypotheses ------------------------------------
md(r"""## 2.5 Exploratory Insight and Hypotheses

The explore-data task moves beyond surface description and quality checks toward the relationships and sub-population structure that will shape the model, and its explicit purpose is to form hypotheses that later phases will test. The analyses in this subsection are deliberately confined to what can be established without a trained model, so that they remain genuine data understanding rather than premature evaluation. Three questions that require a classifier to answer, whether the two health districts are separable to the point that a model trained on one fails on the other, whether the pattern of laboratory missingness is itself predictive of disease, and whether patients fall into clean symptom-defined phenotypes, are posed here as hypotheses and answered in Section 5, where a model is available to test them.

The analyses below use the in-memory target and symptom encodings built in Section 2.3; they read the data but do not alter it. A small amount of statistical machinery, a Fisher odds ratio with its p-value and a Benjamini-Hochberg correction for testing many features at once, is defined inline.
""")
code(r'''from scipy import stats

def fisher_or(x, y):
    x = x.fillna(0).astype(int); y = y.astype(int)
    a = int(((x == 1) & (y == 1)).sum()); b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum()); d = int(((x == 0) & (y == 0)).sum())
    orr = (a + .5) * (d + .5) / ((b + .5) * (c + .5))
    _, p = stats.fisher_exact([[a, b], [c, d]])
    return {"OR": orr, "p": p}

def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[idx] * n / r); q[idx] = prev
    return q

site_raw = raw[SITE_COL].str.strip()
print("Site values:", site_raw.value_counts().to_dict())
''')

md(r"""### 2.5.1 The cohort is two populations, not one

The strongest structural fact in the data is that the patient identifier column, the health district, is not an incidental label but a distribution-shift axis. Disease prevalence differs sharply between the two districts: dengue and typhoid are several times more common at one site than the other. The figure below quantifies this with Wilson confidence intervals on each disease rate by site. This is prior-probability shift, the form of distribution shift most damaging to a classifier, because a model that learns the disease frequencies of one site carries the wrong priors to the other. The figure motivates a hypothesis tested in Section 5, that a model can tell the two sites apart almost perfectly and that symptom-only performance collapses when it is required to generalise from one site to the other, and it justifies retaining the site variable through to the fairness and generalisation analyses rather than discarding it as an identifier.
""")
code(r'''from math import sqrt
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)

rows = []
for lbl in TARGET_LABELS:
    y = pd.to_numeric(raw[DISEASE_COLS[lbl]], errors="coerce")
    do = y[site_raw == "CMA de DO"]; da = y[site_raw == "CMA de DAFRA"]
    pdo, ldo, hdo = wilson(int(do.sum()), int(do.notna().sum()))
    pda, lda, hda = wilson(int(da.sum()), int(da.notna().sum()))
    rr = pdo / pda if pda > 0 else np.inf
    rows.append((lbl, f"{pdo:.0%} [{ldo:.0%},{hdo:.0%}]", f"{pda:.0%} [{lda:.0%},{hda:.0%}]", round(rr, 1)))
ipd.display(pd.DataFrame(rows, columns=["disease", "DO (n=147)", "DAFRA (n=152)", "risk ratio DO/DAFRA"]))

def _site_rate_ci(l, s):
    y = pd.to_numeric(raw[DISEASE_COLS[l]], errors="coerce"); m = site_raw == s
    p, lo, hi = wilson(int(y[m].sum()), int(y[m].notna().sum())); return p * 100, (p - lo) * 100, (hi - p) * 100

fig, ax = plt.subplots(figsize=(8, 3.6)); ax.set_axisbelow(True); w = .38; xs = np.arange(len(TARGET_LABELS))
for off, s, col, lab in [(-w / 2, "CMA de DO", SITE_A, "DO (n=147)"), (w / 2, "CMA de DAFRA", SITE_B, "DAFRA (n=152)")]:
    vals = [_site_rate_ci(l, s) for l in TARGET_LABELS]
    rates = [v[0] for v in vals]; err = [[v[1] for v in vals], [v[2] for v in vals]]
    b = ax.bar(xs + off, rates, w, label=lab, color=col, zorder=3)
    ax.errorbar(xs + off, rates, yerr=err, fmt="none", ecolor="#444", elinewidth=0.9, capsize=2.5, zorder=4)
    bar_labels(ax, b, fmt="{:.0f}%", dx=13, dy=3, fs=8)
ax.set_xticks(xs); ax.set_xticklabels([DISP[l] for l in TARGET_LABELS])
style_ax(ax, ylabel="% positive [95% Wilson CI]")
ax.set_title("Strong disease label shift between sites", loc="center", pad=12, fontsize=15)
ax.legend(loc="upper right")
plt.show()
''')

md(r"""### 2.5.2 Laboratory missingness is not random

Whether missing values can be safely filled with a column average depends on the mechanism that produced them. If values were missing completely at random, missingness would be independent of everything else; the test below checks each frequently-missing feature for dependence on patient age and on site. Missingness depends on age or site for a large fraction of these features, so the missing-completely-at-random assumption is rejected and the mechanism is at least missing-at-random. The practical consequence, that a simple mean imputation would introduce bias and that imputation paired with an explicit missing-indicator should instead be fitted inside each cross-validation fold, is a hypothesis about the right preparation strategy that Section 3 acts on and Section 5 confirms. The same finding is the first hint of the missingness-as-signal idea, because a missingness pattern that tracks patient characteristics may also track diagnosis.
""")
code(r'''# Use the in-memory raw-encoded symptom + numeric view; missingness on the frequently-missing columns.
all_num = {nm: pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
           for nm, col in {"age": C[2], "weight": C[3], "MUA": C[4], "temp": C[63], "resp": C[64],
                            "pulse": C[65], "platelets": C[80], "hematocrit": C[73], "wbc": C[79],
                            "neutrophils": C[81], "lymphocytes": C[77], "creatinine": C[83],
                            "BP": None}.items() if col is not None}
age_years = all_num["age"]
# The leak columns identified in Section 2.4.2 are not legitimate features, so they are
# excluded from the missingness-mechanism test just as they are from the analyses above.
LEAK_COLS = [C[97], C[98], C[107]]
miss_cols = {c: raw[c] for c in raw.columns
             if 0.05 < raw[c].isna().mean() < 0.95 and c not in LEAK_COLS}
do_ind = pd.Series((site_raw == "CMA de DO").astype(int).values, index=raw.index)
rows = []
for name, col in miss_cols.items():
    m = col.isna().astype(int)
    if m.sum() > 1 and (m == 0).sum() > 1:
        p_age = stats.mannwhitneyu(age_years[m == 1].dropna(), age_years[m == 0].dropna()).pvalue
    else:
        p_age = 1.0
    p_site = fisher_or(do_ind, m)["p"]
    rows.append((name, col.isna().mean(), p_age, p_site))
G = pd.DataFrame(rows, columns=["feature", "miss_rate", "p_age", "p_site"])
G["q_age"] = bh_fdr(G["p_age"].fillna(1)); G["q_site"] = bh_fdr(G["p_site"].fillna(1))
G["MAR"] = (G["q_age"] <= .05) | (G["q_site"] <= .05)
print(f"MCAR test over {len(G)} frequently-missing columns: "
      f"{int(G['MAR'].sum())} depend on age or site (FDR q <= 0.05)")
print(f"  -> MCAR {'rejected' if G['MAR'].mean() > 0.3 else 'not rejected'}; "
      f"the mechanism is at least missing-at-random.")
ipd.display(G.sort_values('miss_rate', ascending=False).head(8).round(4))
''')

md(r"""The test above reports which fields depend on age or site, but it does not show the shape of the missingness, and the shape is where the missingness-as-signal idea becomes visible. The interactive figure below presents that shape in two linked panels. The left panel is a nullity matrix in which every row is a patient and every column a laboratory or vital field, with a filled cell marking a missing value and the rows arranged so that similar patterns of missingness sit together; the horizontal bands that emerge separate patients who were investigated very little from patients who were investigated thoroughly, and that ordering is the signal the later phases test for diagnostic content. The right panel is a co-missingness map in which two fields are joined more strongly when they tend to be absent together, so the blocks that appear are the panels of tests that a clinician orders or skips as a unit. Hovering either panel reveals the underlying value, which makes the figure an instrument for inspecting individual patterns rather than only a summary.
""")
code(r'''# Per-patient missingness of the frequently-missing lab and vital fields, over the cohort.
from scipy.cluster.hierarchy import linkage, leaves_list
LabM = (pd.DataFrame({nm: all_num[nm] for nm in
                      ["platelets","hematocrit","wbc","neutrophils","lymphocytes",
                       "creatinine","MUA","weight","resp","pulse","temp"] if nm in all_num})
        .reset_index(drop=True)[labelled.values].reset_index(drop=True))
_mr = LabM.isna().mean()
LabM = LabM[list(_mr[(_mr > 0.05) & (_mr < 0.98)].index)]
Miss = LabM.isna().astype(int)
col_order = Miss.mean().sort_values(ascending=False).index.tolist()
Mo = Miss[col_order]
row_order = leaves_list(linkage(Mo.values, method="ward")) if Mo.values.sum() > 0 and len(Mo) > 2 else np.arange(len(Mo))
Z = Mo.iloc[row_order]
corr_m = Miss[col_order].corr().fillna(0)

from plotly.subplots import make_subplots
figm = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.12,
                     subplot_titles=("Nullity matrix (dark = missing)", "Co-missingness correlation"))
figm.add_trace(go.Heatmap(z=Z.values, x=col_order, y=list(range(len(Z))),
    colorscale=[[0, "#F2F4F7"], [1, "#3D5A80"]], showscale=False,
    customdata=np.where(Z.values == 1, "MISSING", "present"),
    hovertemplate="patient row %{y}<br>%{x}<br>%{customdata}<extra></extra>"), row=1, col=1)
figm.add_trace(go.Heatmap(z=corr_m.values, x=col_order, y=col_order, zmin=-1, zmax=1,
    colorscale="RdBu_r", colorbar=dict(title="corr", len=0.8, x=1.02),
    hovertemplate="%{y} & %{x}<br>missingness corr = %{z:.2f}<extra></extra>"), row=1, col=2)
figm.update_yaxes(showticklabels=False, title_text=f"patients (n={len(Z)})", row=1, col=1)
figm.update_xaxes(tickangle=-45, row=1, col=1); figm.update_xaxes(tickangle=-45, row=1, col=2)
figm.update_yaxes(autorange="reversed", row=1, col=2)
show_interactive(figm, "fig_missingness_structure", height=520, width=1000,
                 title="Laboratory missingness has structure, not noise")
print("Per-patient missing-field count: range "
      f"{int(LabM.isna().sum(1).min())}-{int(LabM.isna().sum(1).max())}, "
      f"median {int(LabM.isna().sum(1).median())}.")
''')

md(r"""### 2.5.3 The malaria base-rate trap

Because malaria is present in ninety percent of patients, the group without dengue is almost entirely malaria, which distorts any naive association between a symptom and dengue. A symptom that marks malaria will appear protective against dengue simply because the dengue-negative group is saturated with malaria, an artefact of the base rate rather than a real effect. The honest quantity is the information a symptom carries about dengue conditional on malaria status. The analysis below ranks symptoms by this malaria-adjusted information and flags the base-rate artefacts, the symptoms that look dengue-protective only because they track malaria. This conditioning is the reason the modelling phase resolves malaria first in the label chain.
""")
code(r'''def entropy(c): c = c / c.sum(); c = c[c > 0]; return -(c * np.log2(c)).sum()
def mi(x, y):
    ct = pd.crosstab(x, y).values.astype(float)
    if ct.sum() == 0: return 0.0
    return max(0.0, entropy(ct.sum(1)) + entropy(ct.sum(0)) - entropy(ct.flatten()))
def cmi_given(x, y, z):
    out = 0.0
    for zv in np.unique(z):
        m = z == zv
        if m.sum() > 2: out += m.mean() * mi(x[m], y[m])
    return out

mal = Y["malaria"].values
dengue_v = Y["dengue"].values
rank = [s for s in SYM if Xsym[s].sum() >= 10]
rows = [(s, mi(pd.Series(Xsym[s].fillna(0).astype(int).values), pd.Series(dengue_v)),
         cmi_given(pd.Series(Xsym[s].fillna(0).astype(int).values), pd.Series(dengue_v), mal)) for s in rank]
MI = pd.DataFrame(rows, columns=["symptom", "MI", "CMI_given_malaria"]).sort_values("CMI_given_malaria", ascending=False)
print("Top dengue symptoms by information conditional on malaria (the malaria-adjusted signal):")
print(MI.head(8).round(4).to_string(index=False))
arte = [s for s in rank if fisher_or(Xsym[s].fillna(0), pd.Series(dengue_v))["OR"] < 0.8
        and fisher_or(Xsym[s].fillna(0), pd.Series(mal))["OR"] > 1.2]
print(f"\nBase-rate artefacts (malaria-enriched but appear dengue-protective): {arte[:8]}")
''')

md(r"""The trap and its correction are clearest as a pair of pictures. The left panel places each symptom by its association with malaria on the horizontal axis and with dengue on the vertical axis, both on a log-odds scale. The symptoms in the lower-right quadrant, which mark malaria yet appear protective against dengue, are the base-rate artefacts a naive one-versus-rest reading would be misled by. The right panel ranks the symptoms by their malaria-adjusted information about dengue and draws a connecting segment from each symptom's raw, unadjusted value to its adjusted value, so that the size and direction of the correction are visible as the length of the segment rather than hidden in two near-coincident points. A long segment marks a symptom whose apparent dengue signal changes substantially once malaria is held fixed, which is exactly the base-rate distortion this analysis is meant to expose. The malaria-adjusted shortlist, rather than the raw associations, is what should inform the model.
""")
code(r'''fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# (a) base-rate trap quadrant: log2 OR vs malaria (x) against log2 OR vs dengue (y)
xs_ = np.array([np.log2(max(fisher_or(Xsym[s].fillna(0), pd.Series(mal))["OR"], 1e-3)) for s in rank])
yd_ = np.array([np.log2(max(fisher_or(Xsym[s].fillna(0), pd.Series(dengue_v))["OR"], 1e-3)) for s in rank])
is_art = np.array([(yd_[k] < np.log2(0.8)) and (xs_[k] > np.log2(1.2)) for k in range(len(rank))])
axes[0].set_axisbelow(True)
axes[0].scatter(xs_[~is_art], yd_[~is_art], s=42, color=SECONDARY, alpha=0.7,
                edgecolor="white", linewidth=0.5, label="ordinary symptom", zorder=3)
axes[0].scatter(xs_[is_art], yd_[is_art], s=90, color=NEGATIVE, edgecolor="white",
                linewidth=1, label="base-rate artefact", zorder=4)
axes[0].axhline(0, color="#bbb", lw=1); axes[0].axvline(0, color="#bbb", lw=1)
for k in np.where(is_art)[0]:
    short = re.sub(r"\s*\(.*?\)\s*$", "", str(rank[k]))[:18]
    axes[0].annotate(short, (xs_[k], yd_[k]), xytext=(5, -2), textcoords="offset points",
                     fontsize=7, color=NEGATIVE, va="top")
style_ax(axes[0], xlabel="log2 odds ratio vs malaria", ylabel="log2 odds ratio vs dengue", grid_axis="both")
axes[0].set_title("The base-rate trap", loc="center", pad=10, fontsize=13)
axes[0].legend(loc="upper left", fontsize=8)

# (b) raw MI vs malaria-adjusted CMI, drawn as a shift: a leader segment from the raw
# value to the adjusted value makes the size and direction of the correction legible,
# rather than two near-coincident points that look identical.
top = MI.sort_values("CMI_given_malaria", ascending=False).head(8)[::-1]
yidx = np.arange(len(top))
raw_v = top["MI"].values
adj_v = top["CMI_given_malaria"].values
axes[1].set_axisbelow(True)
# leader segments: green where conditioning raises the signal, amber where it lowers it
for k in range(len(top)):
    up = adj_v[k] >= raw_v[k]
    axes[1].plot([raw_v[k], adj_v[k]], [yidx[k], yidx[k]],
                 color=(POSITIVE if up else WARNING), lw=2.4, alpha=0.7, zorder=2,
                 solid_capstyle="round")
axes[1].scatter(raw_v, yidx, color=SECONDARY, s=46, zorder=4, edgecolor="white",
                linewidth=0.6, label="raw (unadjusted)")
axes[1].scatter(adj_v, yidx, color=DCOL["dengue"], s=64, zorder=5, edgecolor="white",
                linewidth=0.6, label="adjusted for malaria")
axes[1].set_yticks(yidx)
axes[1].set_yticklabels([re.sub(r"\s*\(.*?\)\s*$", "", str(s))[:22] for s in top["symptom"]], fontsize=8)
style_ax(axes[1], xlabel="information about dengue (bits)", grid_axis="x")
axes[1].set_xlim(left=0)
axes[1].set_title("Raw signal shifts once malaria is held fixed", loc="center", pad=10, fontsize=13)
axes[1].legend(loc="lower right", fontsize=8)

fig.suptitle("Conditioning on malaria reorders the honest dengue signal",
             y=1.0, fontsize=15, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
''')

md(r"""### 2.5.4 Age structure and statistical power

Two final checks set honest expectations for what the data can support. Disease risk is not flat across age: dengue and typhoid climb steeply with age band, which argues for an explicit age feature and warns that age will be a likely axis of unfairness to audit. The power calculation is the more sobering of the two. At the size of the rare-disease cells, only very large effects are detectable at all, which means that any null result for yellow fever or typhoid must be read as underpowered rather than as evidence of no effect. This is a guardrail on interpretation that the whole project carries forward.
""")
code(r'''bands = pd.cut(age_years[labelled.values].reset_index(drop=True), [0, 5, 15, 200], labels=["U5", "5-15", "15+"])
Yr = Y.reset_index(drop=True)
rate_tab = pd.concat([Yr, bands.rename("band")], axis=1).groupby("band", observed=True)[["malaria", "dengue", "typhoid"]].mean()
print("Disease rate by age cohort:")
print(rate_tab.round(2).to_string())
codes = bands.cat.codes.values
for l in ["dengue", "typhoid"]:
    tab = pd.crosstab(codes, Yr[l]); sc = np.arange(tab.shape[0]); col = tab.values; N = col.sum()
    r = col[:, 1]; nr = col.sum(1); pb = r.sum() / N; T = np.sum(sc * (r - nr * pb))
    v = pb * (1 - pb) * (np.sum(nr * sc ** 2) - (np.sum(nr * sc)) ** 2 / N); z = T / np.sqrt(v) if v > 0 else 0
    print(f"  {DISP[l]} age trend: z={z:+.2f}  p={2 * (1 - stats.norm.cdf(abs(z))):.3g}")

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
an = NormalIndPower()
def mdor(npos, nneg, base=.4, alpha=.05, power=.8):
    for orr in np.arange(1.1, 12, .1):
        odds = base / (1 - base) * orr; p2 = odds / (1 + odds)
        try:
            pw = an.power(effect_size=proportion_effectsize(p2, base), nobs1=npos, alpha=alpha, ratio=nneg / max(1, npos))
        except Exception:
            pw = 0
        if pw >= power:
            return round(orr, 1)
    return np.nan
print("\nMinimum detectable odds ratio at the rare-disease cells (80% power):")
for nm, npos in [("yellow fever (n=12)", 12), ("non-malarial dengue (n=14)", 14), ("typhoid (n=29)", 29)]:
    print(f"  {nm:28s}: OR >= {mdor(npos, 299 - npos)}")
print("At n = 12-14 only very large effects are detectable; rare-disease nulls are underpowered, not negative.")
''')

md(r"""Both checks are more legible as figures. The left panel traces the disease rate across the three age bands, where the steep rise of dengue and typhoid with age is immediate. The right panel plots the smallest odds ratio that the data could detect at eighty percent power against the number of positive cases in a cell, with the rare-disease cells marked; it shows concretely that at twelve to fourteen cases only very large effects are within reach, which is why a null for those diseases is reported as underpowered rather than as a finding.
""")
code(r'''fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))

# (a) disease rate by age band
axes[0].set_axisbelow(True)
band_order = ["U5", "5-15", "15+"]
for l in ["malaria", "dengue", "typhoid"]:
    axes[0].plot(band_order, [rate_tab.loc[b, l] * 100 for b in band_order], marker="o", lw=2,
                 color=DCOL[l], label=DISP[l])
d15 = rate_tab.loc["15+", "dengue"] * 100
axes[0].annotate("dengue rises\nsharply (p<1e-7)", (2, d15), xytext=(2, d15 + 22),
                 ha="center", fontsize=8, color=DCOL["dengue"],
                 arrowprops=dict(arrowstyle="->", color=DCOL["dengue"]))
style_ax(axes[0], xlabel="age band", ylabel="% positive")
axes[0].set_title("Disease risk is age-structured", loc="center", pad=10, fontsize=13)
axes[0].legend(loc="center left")

# (b) power floor: minimum detectable odds ratio vs cell size
axes[1].set_axisbelow(True)
ns = np.arange(8, 80)
mdors = [mdor(n, 299 - n) for n in ns]
axes[1].plot(ns, mdors, color=PRIMARY, lw=2, zorder=3)
axes[1].fill_between(ns, mdors, 12, color=NEGATIVE, alpha=0.06)
lab_pos = {"YF (n=12)": (12, (28, 7.2)), "non-mal. dengue (n=14)": (14, (30, 5.6)), "typhoid (n=29)": (29, (44, 4.0))}
for nm, (npos, (tx, ty)) in lab_pos.items():
    my = mdor(npos, 299 - npos)
    axes[1].scatter([npos], [my], color=NEGATIVE, zorder=5)
    axes[1].annotate(nm, (npos, my), xytext=(tx, ty), fontsize=8, color="#444", va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", color="#BBB", lw=0.8))
style_ax(axes[1], xlabel="positives in cell", ylabel="min detectable OR (80% power)")
axes[1].set_title("What N=299 can detect", loc="center", pad=10, fontsize=13)
axes[1].text(58, 9.2, "only very large\neffects detectable", color=NEGATIVE, fontsize=10, ha="center")

fig.suptitle("Rare diseases are real but underpowered at N=299", y=1.0, fontsize=15, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
''')

md(r"""### 2.5.5 How the symptoms group together

The single-symptom ceiling in Section 2.3 showed that no individual sign separates dengue, but it left open whether the signs combine into coherent groups. The interactive figure below answers that without a model. Each node is a symptom and two symptoms are joined when they co-occur more often than chance, measured by the phi coefficient, so that tightly linked signs are drawn close and a community-detection step colours the groups that emerge. The figure deliberately carries no fixed labels and no side panel; hovering a node names the sign, the syndrome it belongs to, the dengue enrichment of that syndrome, and how many other signs it accompanies, and a syndrome can be isolated by clicking it in the compact legend. Read this way the picture shows that the symptoms do assemble into clinically recognisable syndromes, a generic febrile group, a haemorrhagic group, a respiratory group, a neurological group, and a small liver-enzyme group, yet only the liver-enzyme group is appreciably enriched for dengue. The structure is real but it does not hand the model a single discriminating syndrome, which reinforces from the multivariate side the conclusion that dengue must be learned from a combination of signs rather than from any one of them.
""")
code(r'''# Symptom co-occurrence network over the cohort (model-free; from Xsym and Y).
import networkx as nx
Xs5 = Xsym.fillna(0).astype(int)
def _short(name, k=20):
    s = fold_raw(name).title()
    if len(s) <= k:
        return s
    cut = s[:k].rsplit(" ", 1)[0]      # cut on a word boundary, no dangling ellipsis
    return cut if len(cut) >= 6 else s[:k]
SYM_SHORT = {c: _short(c) for c in Xs5.columns}

def _phi(a, b):
    A = Xs5[a].values; B = Xs5[b].values; n = len(A)
    s1 = A.sum(); s2 = B.sum(); s12 = ((A == 1) & (B == 1)).sum()
    den = np.sqrt(s1 * (n - s1) * s2 * (n - s2)) / n
    return (s12 - s1 * s2 / n) / den if den > 0 else 0.0

cols5 = [c for c in Xs5.columns if Xs5[c].sum() >= 15]
GR = nx.Graph()
for c in cols5: GR.add_node(c, prev=Xs5[c].mean())
TH = 0.30
for i in range(len(cols5)):
    for j in range(i + 1, len(cols5)):
        w = _phi(cols5[i], cols5[j])
        if w >= TH: GR.add_edge(cols5[i], cols5[j], weight=w)
GR.remove_nodes_from([n for n in list(GR.nodes) if GR.degree(n) == 0])

import networkx.algorithms.community as nxcom
comms5 = sorted([c for c in nxcom.greedy_modularity_communities(GR, weight="weight") if len(c) >= 2],
                key=len, reverse=True)
node_comm = {n: ci for ci, com in enumerate(comms5) for n in com}
wdeg = dict(GR.degree(weight="weight"))
hubs = {ci: max(com, key=lambda n: wdeg.get(n, 0)) for ci, com in enumerate(comms5)}
den_v5 = Y["dengue"].reset_index(drop=True).values
def _lift(members):
    idx = Xs5[list(members)].max(1).values == 1
    return (den_v5[idx].mean() + 1e-9) / (den_v5.mean() + 1e-9) if idx.sum() else 1.0
syn = {ci: f"{SYM_SHORT[hubs[ci]]} (x{_lift(com):.1f} dengue)" for ci, com in enumerate(comms5)}

posg = nx.kamada_kawai_layout(GR, weight="weight")
palette = [PRIMARY, NEGATIVE, POSITIVE, WARNING, "#7B5EA7", "#00897B", "#B0833B"]
ex, ey = [], []
for u, v, d in GR.edges(data=True):
    ex += [posg[u][0], posg[v][0], None]; ey += [posg[u][1], posg[v][1], None]
edge_tr = go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#CBD2DA", width=1.0),
                     hoverinfo="skip", showlegend=False)
node_trs = []
for ci, com in enumerate(comms5):
    members = [n for n in com if n in GR.nodes]
    nx_ = [posg[n][0] for n in members]; ny_ = [posg[n][1] for n in members]
    sizes = [12 + 46 * GR.nodes[n]["prev"] for n in members]
    txt = [f"<b>{SYM_SHORT[n]}</b><br>syndrome: {syn[ci]}<br>co-occurs with {GR.degree(n)} signs"
           f"<br>prevalence {GR.nodes[n]['prev']:.0%}" for n in members]
    node_trs.append(go.Scatter(x=nx_, y=ny_, mode="markers", name=syn[ci],
        marker=dict(size=sizes, color=palette[ci % len(palette)], line=dict(color="white", width=1.2)),
        text=txt, hovertemplate="%{text}<extra></extra>"))
fign = go.Figure([edge_tr] + node_trs)
fign.update_xaxes(visible=False); fign.update_yaxes(visible=False)
fign.update_layout(legend=dict(orientation="v", x=1.0, y=1.0, font=dict(size=10),
                               title=dict(text="syndrome (click to toggle)", font=dict(size=10))),
                   hoverlabel=dict(bgcolor="white"))
show_interactive(fign, "fig_symptom_network", height=640, width=940,
                 title="Symptoms cluster into syndromes, but none cleanly marks dengue")
print(f"Symptom network: {GR.number_of_nodes()} signs, {len(comms5)} syndromes (phi >= {TH}).")
''')

# --- 2.6 Data Understanding Summary -------------------------------------------
md(r"""## 2.6 Data Understanding Summary

The data-quality task closes by listing the problems found together with the solutions they call for, and the explore-data task closes by recording the hypotheses it has raised for later phases. Both lists are gathered here so that Section 3 receives an explicit mandate and Section 5 inherits a clear set of claims to test. Nothing in this section has altered the cohort; the table below is the contract for the changes that Section 3 will make.

**Data-quality findings and the repairs deferred to Data Preparation.**

| Finding (detected in 2.4) | Evidence | Repair, applied in Section 3 |
|---|---|---|
| Impossible numeric entries (temperature, blood pressure, pulse) | Values outside survivable physiology; raw strings show dropped decimals and run-together readings | Repair each value from itself within wide physiological bounds; null only the genuinely impossible |
| A feature copies the dengue label | *Dengue (Dengua)* positive maps onto dengue with an empty off-diagonal cell | Exclude this column, the concatenated-diagnosis column, and the free-text field from the feature matrix |
| Laboratory units are not comparable | Platelets split into two clean unit clusters; the other labs smear across four to five decades | Reconcile the platelet units; tag the remaining labs as rank-only and draw no absolute-level inference |
| Redundant and unlabelled records | Headache pair carries different signal; urination pair agrees almost perfectly; one row has no label | Keep both headache columns; merge the urination pair with a missing-indicator; drop the unlabelled row, giving 299 patients |

**Hypotheses raised for Modeling and Evaluation.**

| Hypothesis (raised in 2.3 and 2.5) | Where it is tested |
|---|---|
| The label space is too sparse for a single class over combinations; related binary labels with malaria first are required | Section 4 (Modeling) |
| The two sites are almost perfectly separable, and symptom-only performance collapses when generalising across sites | Section 5 (Evaluation), with site-aware validation |
| The laboratory missingness pattern is itself predictive of disease once leakage is controlled | Section 4 and Section 5 |
| Patients do not form clean symptom-defined phenotypes, so a probabilistic multi-label model is preferable to rules or clustering | Section 5 |
| Rare-disease results are underpowered, so any null for yellow fever or typhoid is reported descriptively | Throughout |

With the data described, its target understood, its defects catalogued, and its deeper structure turned into testable hypotheses, the preparation phase can now construct the leak-free dataset that the model will use.
""")

# ============================================================================
# SECTION 3 — DATA PREPARATION
# ============================================================================
md(r"""---

# 3. Data Preparation

> **Purpose of this phase (CRISP-DM).** Data preparation covers all the activities that construct the final dataset, the data that will be fed to the model, from the initial raw data. Its tasks are to select the data to be used, to clean it, to construct any derived attributes, to integrate sources where needed, and to format it for the modelling tools (Chapman et al., 2000). It is typically the most laborious phase of a data-mining project, and it is performed iteratively rather than in a single pass.

This section executes the contract drawn up in Section 2.6. Every repair and construction below traces back to a specific finding from Data Understanding, and the cross-references make that lineage explicit so that no transformation is unexplained. Two disciplines govern the work. First, only transformations that cannot leak information are performed here. A repair is admissible at this stage when it depends on the value being repaired alone, with no statistic computed across patients, because such a row-local operation cannot transfer information between the training and test portions of a later cross-validation. Second, every transformation that *would* require a statistic estimated from the data, imputation of missing values, scaling of continuous features, and the rank transformation of the non-comparable laboratory fields, is deliberately deferred so that it can be fitted inside each cross-validation fold during modelling. The output of this section is therefore a clean, leak-free feature matrix together with its multi-label target, its missingness mask, and the site variable, all aligned to the final cohort, but with the data-dependent transformations left for the modelling phase to apply correctly.

The subsections follow the CRISP-DM preparation tasks in order: selection of the data to keep and to exclude (3.1), cleaning of the corrupted values (3.2), construction of the feature matrix including the features motivated by the exploratory findings (3.3), finalisation of the target and cohort (3.4), integration of the site variable (3.5), construction of the missingness mask (3.6), a readiness validation gate (3.7), and a closing summary of what has been produced and what remains deferred (3.8).
""")

# --- 3.1 Data Selection -------------------------------------------------------
md(r"""## 3.1 Data Selection

The selection task decides which columns enter the modelling dataset and which are excluded, with the rationale for each exclusion. Four groups of columns are removed by construction. The two identifier columns carry no clinical signal. The disease one-hot columns are the prediction target and must not appear among the features. The three leakage columns identified in Section 2.4.2, the dengue label-copy, the concatenated-diagnosis string, and the free-text field that names the diagnosis, are excluded because training on them would reconstruct the answer. Finally, any column with a single distinct value carries no information and is dropped. The remaining columns are sorted into the families that determine how each is encoded in Section 3.3. The household-dengue history field and the two confirmatory-test columns are set aside from the automatic sorting because they are constructed explicitly later, which avoids building them twice.
""")
code(r'''import unicodedata, re
def _fold(s):
    """Strip accents and upper-case, robust to composed/decomposed Unicode ('Negatif ' -> 'NEGATIF')."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).strip().upper()

LEAK_OR_LABEL_COLS = [LEAK_COL_DENGUE, COMBINED_DX_COL, C[107]]
disease_set = set(DISEASE_COLS.values())
HANDLED_EXPLICITLY = {HOUSEHOLD_DENGUE, C[68], C[69]}   # built by hand in 3.3

binary_symptom_cols, posneg_cols, numeric_like_cols, other_cols, zero_var_cols = [], [], [], [], []
for col in raw.columns:
    if col in ID_COLS or col in disease_set or col in LEAK_OR_LABEL_COLS or col in HANDLED_EXPLICITLY:
        continue
    if raw[col].nunique(dropna=True) <= 1:
        zero_var_cols.append(col); continue
    vals = set(_fold(v) for v in raw[col].dropna().unique())
    if vals <= {"OUI", "NON"}:
        binary_symptom_cols.append(col)
    elif vals <= {"POSITIF", "NEGATIF"}:
        posneg_cols.append(col)
    else:
        coerced = pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        (numeric_like_cols if coerced.notna().mean() > 0.6 else other_cols).append(col)

print("Columns excluded by construction:")
print(f"  identifiers          : {len(ID_COLS)}")
print(f"  disease target block  : {len(disease_set)}")
print(f"  leak / label columns  : {len(LEAK_OR_LABEL_COLS)} -> {[c[:24] for c in LEAK_OR_LABEL_COLS]}")
print(f"  zero-variance columns : {len(zero_var_cols)} -> {[c[:22] for c in zero_var_cols]}")
print("\nColumns retained, by family:")
print(f"  binary OUI/NON symptoms : {len(binary_symptom_cols)}")
print(f"  numeric-like            : {len(numeric_like_cols)}")
print(f"  other (categorical/mixed): {len(other_cols)}")
print(f"  handled explicitly in 3.3: household-dengue + 2 confirmatory tests")
''')

# --- 3.2 Data Cleaning --------------------------------------------------------
md(r"""## 3.2 Data Cleaning

The cleaning task repairs the corrupted values detected in Section 2.4.1. Each repair is stateless, acting on a single value without reference to any other patient, which keeps it leak-free. The corruption was shown to be mechanical, so the repairs reverse the specific mechanisms observed: a temperature recorded with a dropped decimal point has its decimal shifted back into the physiological range, a blood-pressure reading run together into one number or written in centimetre-of-mercury shorthand is split and rescaled, an impossible pulse or capillary-refill time is nulled, and an age recorded as a fraction of a year is resolved to a decimal. The physiological bounds are deliberately wide, so that a genuine febrile extreme is never trimmed; a value is nulled only when it is outside the range of human survivability. The cell below defines the repair functions and demonstrates each on the exact corrupted strings found in Section 2.4.1.
""")
code(r'''TEMP_LO, TEMP_HI = 34.0, 43.0
def fix_temperature(val):
    if pd.isna(val): return np.nan
    s = re.sub(r"[^0-9.]", "", str(val).strip().replace(",", "."))
    if s in ("", "."): return np.nan
    try: x = float(s)
    except ValueError: return np.nan
    for _ in range(4):
        if x <= TEMP_HI: break
        x /= 10.0
    return x if TEMP_LO <= x <= TEMP_HI else np.nan

PULSE_LO, PULSE_HI = 30.0, 220.0
def fix_pulse(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if "/" in s: s = s.split("/")[0]
    x = pd.to_numeric(s, errors="coerce")
    return float(x) if pd.notna(x) and PULSE_LO <= x <= PULSE_HI else np.nan

REFILL_LO, REFILL_HI = 0.0, 10.0
def fix_capillary_refill(val):
    if pd.isna(val): return np.nan
    x = pd.to_numeric(str(val).replace(",", "."), errors="coerce")
    return float(x) if pd.notna(x) and REFILL_LO <= x <= REFILL_HI else np.nan

def parse_blood_pressure(val):
    if pd.isna(val): return (np.nan, np.nan)
    s = str(val).strip().replace("|", "/").replace(" ", "").replace(",", ".")
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2: return (np.nan, np.nan)
        sys = pd.to_numeric(parts[0], errors="coerce"); dia = pd.to_numeric(parts[1], errors="coerce")
        if pd.notna(sys) and pd.notna(dia) and sys < 26 and dia < 26:
            sys, dia = sys * 10, dia * 10
    else:
        digits = re.sub(r"[^0-9]", "", s)
        if len(digits) in (5, 6): sys, dia = float(digits[:3]), float(digits[3:])
        elif len(digits) == 4: sys, dia = float(digits[:2]), float(digits[2:])
        else:
            v = pd.to_numeric(s, errors="coerce"); sys, dia = (v, np.nan) if pd.notna(v) else (np.nan, np.nan)
    sys = sys if (pd.notna(sys) and 50 <= sys <= 260) else np.nan
    dia = dia if (pd.notna(dia) and 20 <= dia <= 160) else np.nan
    return (float(sys) if pd.notna(sys) else np.nan, float(dia) if pd.notna(dia) else np.nan)

def fix_age(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(",", ".")
    if "/" in s:
        num, den = s.split("/")
        try: return float(num) / float(den)
        except (ValueError, ZeroDivisionError): return np.nan
    return pd.to_numeric(s, errors="coerce")

print("temperature:", {v: (round(fix_temperature(v), 3) if pd.notna(fix_temperature(v)) else None)
                        for v in ["3909", "390", "T3501", "367", "36,6", "19.76"]})
print("blood press:", {v: parse_blood_pressure(v) for v in ["10976", "104|58", "13/8", "98"]})
print("age        :", {v: round(fix_age(v), 3) for v in ["7/12", "15/12", "3,83", "57"]})
print("pulse      :", {v: fix_pulse(v) for v in [124000, "112/58", 92]})
print("cap refill :", {v: fix_capillary_refill(v) for v in [2.0, 29.0, 385.0, 3.0]})
print("accent fold:", _fold("Negatif "))
''')

# --- 3.3 Feature Construction -------------------------------------------------
md(r"""## 3.3 Feature Construction

The construction task assembles the cleaned values into a single feature matrix and builds the derived attributes. Several decisions made here implement findings from Data Understanding rather than arbitrary engineering choices. The binary symptoms are encoded from their French *OUI* and *NON*, and the two confirmatory tests from *Positif* and *Negatif*, using the accent-safe folding that was shown in Section 2.1 to be necessary. The platelet count is reconciled to a single unit by multiplying its lower cluster by a thousand, the one laboratory repair that Section 2.4.3 found to be safe, while the remaining laboratory fields are tagged as ambiguous so that the modelling phase rank-transforms them and draws no inference from their absolute levels. The two headache columns are both retained, because Section 2.4.4 showed they carry different signal, and the two frequent-urination columns are merged, because Section 2.4.4 showed they are genuine duplicates; the merge preserves a missing value only where both sources are missing, so that the missingness-as-signal analysis still sees it.

Two derived features are added that are motivated directly by the exploratory findings of Section 2.5. A symptom-burden count, the number of symptoms a patient reports, is constructed because it is a cheap and robust summary that tracks the number of co-occurring diagnoses. An age band is constructed because Section 2.5.4 found that dengue and typhoid risk climbs steeply with age; encoding age into the under-five, five-to-fifteen, and over-fifteen groups gives the model an explicit handle on that gradient and names the axis along which the fairness audit will later look for disparities. Because both of these are functions of other features rather than raw measurements, they are placed in a dedicated derived group rather than among the symptoms or the demographics. This keeps the later feature-regime comparison honest, since a symptoms-only regime must exclude a count built from the symptoms, and a laboratory-free regime must keep an age band that is in no sense a laboratory field. Each feature is registered under a unique, readable handle and recorded in its group, and a guard asserts that the columns whose values depend on correct accent decoding are not constant, which is the failure that an encoding mistake would produce.
""")
code(r'''def _ascii_snake(text, maxlen=32):
    text = "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text[:maxlen] or "feat"

NAME_OVERRIDES = {
    C[44]: "delirium", C[59]: "moderate_splenomegaly", C[62]: "lymphadenopathy",
    C[70]: "hemoconcentration", C[84]: "alat_asat_elevated", C[85]: "diabetes",
    C[86]: "hypertension", C[87]: "sickle_cell", C[90]: "osteoarthritis",
}
def short_handle(col):
    if col in NAME_OVERRIDES: return NAME_OVERRIDES[col]
    m = re.search(r"\(([^)]*)\)\s*$", col)
    gloss = _ascii_snake(m.group(1)) if m else ""
    return gloss or _ascii_snake(re.sub(r"\s*\([^)]*\)\s*$", "", col))

def encode_oui_non(series):
    return series.map(lambda v: {"OUI": 1.0, "NON": 0.0}.get(_fold(v), np.nan)
                      if pd.notna(v) else np.nan).astype("float")
def encode_pos_neg(series):
    return series.map(lambda v: {"POSITIF": 1.0, "NEGATIF": 0.0}.get(_fold(v), np.nan)
                      if pd.notna(v) else np.nan).astype("float")

feat = {}
feature_groups = {"symptom": [], "vital": [], "lab": [], "lab_ambiguous": [], "demographic": [],
                  "confirmatory_test": [], "history": [], "derived": []}
def add(name, series, group):
    h, i = name, 2
    while h in feat:
        h = f"{name}_{i}"; i += 1
    feat[h] = series; feature_groups[group].append(h); return h

AGE_COL, WEIGHT_COL, GENDER_COL = C[2], C[3], C[1]
TEMP_COL, PULSE_COL, BP_COL = C[63], C[65], C[66]
HISTORY_BIN   = {C[85], C[86], C[87], C[90], C[93], C[95]}
DUP_URINATION = {C[35], C[38]}

# binary symptoms (both headache columns kept; urination pair handled below)
for col in binary_symptom_cols:
    if col in DUP_URINATION: continue
    add(short_handle(col), encode_oui_non(raw[col]), "history" if col in HISTORY_BIN else "symptom")

# urination OR-merge, preserving NaN only where BOTH sources are missing (Section 2.4.4)
_u1, _u2 = encode_oui_non(raw[C[35]]), encode_oui_non(raw[C[38]])
add("frequent_urination", (_u1.fillna(0) + _u2.fillna(0)).clip(0, 1).where(~(_u1.isna() & _u2.isna())), "symptom")
add("household_dengue", encode_oui_non(raw[HOUSEHOLD_DENGUE]), "history")

# confirmatory tests (kept but tagged so the modelling phase can ablate them)
add("tdr_positive", encode_pos_neg(raw[C[68]]), "confirmatory_test")
add("thick_smear_positive", encode_pos_neg(raw[C[69]]), "confirmatory_test")

# demographics
add("age_years", raw[AGE_COL].apply(fix_age), "demographic")
add("is_female", raw[GENDER_COL].map(lambda v: {"FEMME": 1.0, "HOMME": 0.0}.get(_fold(v), np.nan)
                                      if pd.notna(v) else np.nan), "demographic")
add("weight_kg", pd.to_numeric(raw[WEIGHT_COL], errors="coerce"), "demographic")
_mua = raw[C[4]].map(lambda v: pd.to_numeric(str(v).replace(",", "."), errors="coerce"))
add("mua_cm", _mua.where(_mua.between(5, 40)), "demographic")

# vitals (cleaned via the stateless repairs of 3.2)
add("temp_c", raw[TEMP_COL].apply(fix_temperature), "vital")
_rr = pd.to_numeric(raw[C[64]].astype(str).str.replace(",", "."), errors="coerce")
add("resp_rate", _rr.where(_rr.between(8, 80)), "vital")
add("pulse_bpm", raw[PULSE_COL].apply(fix_pulse), "vital")
_bp = raw[BP_COL].apply(parse_blood_pressure)
add("bp_systolic", _bp.apply(lambda t: t[0]), "vital")
add("bp_diastolic", _bp.apply(lambda t: t[1]), "vital")
add("capillary_refill_s", raw[C[67]].apply(fix_capillary_refill), "vital")
add("fever_type_recurrent", raw[C[8]].map(
    lambda v: {"RECURRENTE": 1.0, "INTERMITTENTE": 0.0}.get(_fold(v), np.nan) if pd.notna(v) else np.nan), "vital")

# platelets: the one lab with a recoverable single unit (Section 2.4.3); low cluster x1000 -> per uL
_plt = pd.to_numeric(raw[C[80]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
add("platelets", _plt.mask(_plt < 1000, _plt * 1000), "lab")

# the remaining labs pool analytes/units across ~5 decades -> tag ambiguous (rank-transform later)
for name, col in {"hematocrit": C[73], "wbc": C[79], "neutrophils": C[81],
                  "lymphocytes": C[77], "creatinine_elev": C[83]}.items():
    add(name, pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"), "lab_ambiguous")

# derived feature 1: symptom burden (motivated by Section 2.3 / 2.5; tracks co-infection count).
# It is a function of the symptom block rather than a raw field, so it is tagged "derived"
# (not "symptom" and not "demographic") to keep the feature-regime construction in modelling honest.
_sym_so_far = list(feature_groups["symptom"])
add("symptom_burden", pd.DataFrame({h: feat[h] for h in _sym_so_far}).sum(axis=1, min_count=1), "derived")

# derived feature 2: age band (motivated by the age gradient in Section 2.5.4). A deterministic
# function of age_years; tagged "derived" for the same reason as symptom_burden.
_age = raw[AGE_COL].apply(fix_age)
_band = pd.cut(_age, [0, 5, 15, 200], labels=["U5", "5-15", "15+"])
add("age_band", _band.cat.codes.where(_band.notna(), np.nan).astype("float"), "derived")

clean = pd.DataFrame(feat, index=raw.index)
all_handles = [h for hs in feature_groups.values() for h in hs]
assert len(all_handles) == len(set(all_handles)) == clean.shape[1], "feature accounting mismatch"
for col in ["tdr_positive", "thick_smear_positive", "fever_type_recurrent"]:
    assert clean[col].nunique(dropna=True) >= 2, f"'{col}' is constant -> check the file encoding!"

print(f"Assembled {clean.shape[1]} unique features x {clean.shape[0]} rows (accounting OK)")
print("Encoding-sensitive features are non-constant: "
      f"tdr={clean['tdr_positive'].nunique()}, thick_smear={clean['thick_smear_positive'].nunique()}, "
      f"fever_type={clean['fever_type_recurrent'].nunique()}")
print("Feature groups:")
for g, cols in feature_groups.items():
    print(f"  {g:18s}: {len(cols):2d}  e.g. {cols[:4]}")
''')

# --- 3.4 Target and Cohort Finalisation ---------------------------------------
md(r"""## 3.4 Target and Cohort Finalisation

The four-disease target is assembled from the disease one-hot columns, and the cohort is finalised by removing the single record that Section 2.4.4 found to carry no disease label, since a patient with no label cannot supervise a multi-label model. This reduces the cohort to 299 patients. The cell confirms the per-label counts, the co-infection geometry, and the two quantities that anchor the evaluation: the number of non-malarial dengue cases, which is the honest yardstick fixed in Section 2.3, and the degree to which typhoid is nested within malaria.
""")
code(r'''Y_all = pd.DataFrame({lbl: pd.to_numeric(raw[DISEASE_COLS[lbl]], errors="coerce") for lbl in TARGET_LABELS},
                     index=raw.index)
others = pd.to_numeric(raw[DISEASE_COLS["others"]], errors="coerce").rename("others")

all_nan_label = Y_all.isna().all(axis=1)
print(f"Rows with all four labels missing: {all_nan_label.sum()} -> index {list(raw.index[all_nan_label])}")
assert all_nan_label.sum() == 1

keep = ~all_nan_label
X = clean.loc[keep].reset_index(drop=True)
Y = Y_all.loc[keep].astype(int).reset_index(drop=True)
others = others.loc[keep].reset_index(drop=True)
co_count = Y.sum(axis=1)

print(f"\nFinal cohort: N = {len(X)} (one unlabelled row dropped)\n")
print("Per-label positives (N=299):")
for lbl in TARGET_LABELS:
    print(f"  {lbl:13s}: {int(Y[lbl].sum()):3d}  ({Y[lbl].mean()*100:4.1f}%)")
print("\nCo-infection count (diseases per patient):")
print(co_count.value_counts().sort_index().to_string())
print(f"\nNon-malarial dengue (Dengue=1 and Malaria=0): "
      f"{int(((Y['dengue']==1) & (Y['malaria']==0)).sum())}  (the RQ1 yardstick from Section 2.3)")
print(f"Typhoid nested in malaria: {int(((Y['typhoid']==1) & (Y['malaria']==1)).sum())} of {int(Y['typhoid'].sum())}")
''')

# --- 3.5 Site Integration -----------------------------------------------------
md(r"""## 3.5 Integrating the Site Variable

Section 2.5.1 established that the health district is a distribution-shift axis rather than a discardable identifier. It is therefore aligned to the finalised cohort, using the same row removal, and retained alongside the dataset. It is not added to the feature matrix, because it is geography rather than a clinical sign, but it is kept so that the modelling phase can use site-aware cross-validation and the evaluation phase can audit fairness and characterise generalisation across sites. An integrity check confirms the alignment by verifying that the malaria one-hot agrees with the diagnosis string, which guards against any row misalignment introduced by the drop.
""")
code(r'''raw299 = raw.drop(index=int(raw.index[all_nan_label][0])).reset_index(drop=True)
site = raw299[SITE_COL].str.strip().map({"CMA de DO": "DO", "CMA de DAFRA": "DAFRA"}).rename("site")
diag_string = raw299[COMBINED_DX_COL].astype(str).rename("diagnosis_string")
_mal_str = diag_string.str.contains("Paludisme|Malaria", case=False, na=False).astype(int)
assert (_mal_str.values == Y["malaria"].values).mean() > 0.97, "site/row alignment is wrong!"
print(f"site aligned to N={len(site)}: {site.value_counts().to_dict()} (integrity check passed)")
''')

# --- 3.6 Missingness Mask -----------------------------------------------------
md(r"""## 3.6 Constructing the Missingness Mask

The missingness mask is a binary matrix that records, for each value, whether it is present or absent. It is the feature representation of the missingness-as-signal idea raised in Section 2.5.2, and because it is computed row by row it cannot leak. The mask is partitioned into two classes so that the modelling phase can ablate them separately. The generic class records the absence of ordinary clinical measurements, which is legitimate signal about the clinical picture. The confirmatory class records the absence of the two confirmatory tests, which reflects a clinician's decision to order a test and is therefore a potential route for label information to enter; keeping the partition explicit means the confirmatory mask can be removed by construction during the leakage-controlled comparison rather than by memory.
""")
code(r'''M = X.isna().astype(int).add_suffix("__isna")
miss_rate = X.isna().mean().sort_values(ascending=False)

print(f"Mask matrix M: {M.shape[1]} indicators x {M.shape[0]} rows")
print(f"Features fully observed (0% missing): {(miss_rate == 0).sum()} / {len(miss_rate)}")
print(f"Features over 50% missing           : {(miss_rate > 0.50).sum()}")
print("\nMost-missing features (top 12):")
for name, r in miss_rate.head(12).items():
    grp = next(g for g, cols in feature_groups.items() if name in cols)
    print(f"  {name:24s} {r*100:5.1f}%   [{grp}]")

print("\nMean missing rate by feature group:")
for g, cols in feature_groups.items():
    if cols:
        print(f"  {g:18s}: {X[cols].isna().mean().mean()*100:5.1f}%")

CONF_FEATS = list(feature_groups["confirmatory_test"])
mask_confirmatory = [f"{f}__isna" for f in CONF_FEATS if f"{f}__isna" in M.columns]
mask_generic = [c for c in M.columns if c not in mask_confirmatory]
print(f"\nMask partition -> generic: {len(mask_generic)} columns | "
      f"confirmatory (ablated in the leakage-controlled comparison): {mask_confirmatory}")
''')

# --- 3.7 Dataset-Readiness Validation -----------------------------------------
md(r"""## 3.7 Dataset-Readiness Validation

Before the dataset is handed to the modelling phase, a set of hard assertions verifies that the preparation has done what it claimed. Each check is a gate that halts the notebook on failure, which turns the promises made in the previous subsections into guarantees rather than intentions. The checks confirm that the cleaned vitals now lie within physiological bounds, that no leakage column or target label has entered the feature matrix, that the cohort size and the alignment of the feature matrix, the mask, and the site variable are all consistent, that the binary features are genuinely zero-or-one, and that the two derived features are present and sane. This gate is the reason the preparation can be trusted; an error such as a forgotten bound or a misaligned row is caught here rather than discovered as a strange result during modelling.
""")
code(r'''checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", detail)); assert ok, f"VALIDATION FAILED: {name} ({detail})"

phys = {"temp_c": (34, 43), "pulse_bpm": (30, 220), "resp_rate": (8, 80),
        "bp_systolic": (50, 260), "capillary_refill_s": (0, 10), "mua_cm": (5, 40)}
for v, (lo, hi) in phys.items():
    s = X[v].dropna(); frac = ((s >= lo) & (s <= hi)).mean() if len(s) else 1.0
    check(f"{v} within physiologic range", frac == 1.0, f"{frac:.1%} in [{lo},{hi}]")

leaks_present = [c for c in X.columns if "dengua" in c.lower() or "diagnos" in c.lower()]
check("no leak columns in feature matrix", len(leaks_present) == 0, str(leaks_present))
check("target labels absent from features", not (set(TARGET_LABELS) & set(X.columns)), "")

check("cohort size N=299", len(X) == 299, f"N={len(X)}")
check("X and Y row-aligned", (X.index == Y.index).all(), "")
check("mask M aligned to X", M.shape == X.shape and list(M.index) == list(X.index), f"M{M.shape} X{X.shape}")
check("mask partition covers all mask cols", set(mask_generic) | set(mask_confirmatory) == set(M.columns), "")
check("site aligned to N=299", len(site) == 299 and site.isin(["DO", "DAFRA"]).all(), "")
bins = feature_groups["symptom"] + feature_groups["history"] + feature_groups["confirmatory_test"]
nonbin = [c for c in bins if not set(X[c].dropna().unique()) <= {0.0, 1.0}]
check("all binary features are 0/1", len(nonbin) == 0, str(nonbin[:5]))
check("symptom_burden present and sane",
      "symptom_burden" in X.columns and X["symptom_burden"].max() <= len(feature_groups["symptom"]),
      f"max={X['symptom_burden'].max():.0f}")
check("age_band present with 3 levels", "age_band" in X.columns and X["age_band"].nunique(dropna=True) == 3,
      f"levels={int(X['age_band'].nunique(dropna=True))}")

ipd.display(pd.DataFrame(checks, columns=["check", "status", "detail"]))
print(f"\nAll {len(checks)} readiness checks PASSED.")
''')

# --- 3.8 Data Preparation Summary ---------------------------------------------
md(r"""## 3.8 Data Preparation Summary

The preparation phase is complete. Before the closing inventory, the scorecard below condenses the whole remediation into one view: each card is one data-quality issue from Section 2.4, coloured by severity, stating what was found and the concrete repair applied, with the count of records or columns it touched. Read together, the cards make the scale of the cleaning visible at a glance and confirm that every defect catalogued during Data Understanding has a corresponding, traceable action here rather than an unrecorded fix. The most consequential entries are the three leakage columns quarantined from the feature matrix and the single unlabelled patient removed, because those are the actions that most directly protect the honesty of every result that follows.
""")
code(r'''# Data-quality scorecard: one card per Section 2.4 issue, with the real remediation impact
# recomputed from the raw and cleaned data so the figure reports facts rather than a static list.
import matplotlib.patches as mpatches

# --- recompute concrete impact counts -----------------------------------------
_rawnum = lambda col: pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
# a temperature is "corrected" when the physiologic repair differs from the naive numeric read
_temp_repaired = raw[TEMP_COL].apply(fix_temperature)
_temp_naive = _rawnum(TEMP_COL)
n_temp_fixed = int(((_temp_repaired != _temp_naive) & ~(_temp_repaired.isna() & _temp_naive.isna())).sum())
n_leak       = len(LEAK_OR_LABEL_COLS)
n_zerovar    = len(zero_var_cols)
n_labs_amb   = len(feature_groups["lab_ambiguous"])
n_rows_drop  = raw.shape[0] - X.shape[0]
n_plt_rescaled = int((_rawnum(C[80]) < 1000).sum())   # platelet values multiplied to a single unit

SEV = {"high": NEGATIVE, "med": WARNING, "low": POSITIVE}
cards = [
    ("Impossible numeric entries", "med",
     "Temperatures, pulses and blood pressures outside survivable physiology (e.g. 3909, 124000)",
     f"Repaired from each value within wide physiological bounds; impossible values nulled. {n_temp_fixed} temperature cells corrected"),
    ("Label-copy leakage", "high",
     "The 'Dengue (Dengua)' column reconstructs the dengue label with an empty off-diagonal",
     f"{n_leak} leakage or label columns quarantined from the feature matrix"),
    ("Non-comparable lab units", "med",
     "Only platelets split into clean unit clusters; other labs smear across 4 to 5 decades",
     f"Platelets reconciled ({n_plt_rescaled} values rescaled); {n_labs_amb} labs tagged rank-only, no level inference"),
    ("Constant columns", "low",
     "Columns carrying a single distinct value provide no signal",
     f"{n_zerovar} zero-variance columns dropped"),
    ("Redundant discrete fields", "low",
     "Two headache columns carry different signal; the urination pair is a true duplicate",
     "Both headache columns kept; urination pair merged with a preserved missing indicator"),
    ("Unlabelled record", "high",
     "One patient has no entry in any disease column, so its label is unknown",
     f"{n_rows_drop} unlabelled row removed, giving the final cohort of {X.shape[0]}"),
]

fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 3); ax.set_ylim(0, 2); ax.axis("off")
for i, (title, sev, found, fix) in enumerate(cards):
    cx, cy = i % 3, 1 - i // 3
    ax.add_patch(mpatches.FancyBboxPatch((cx + 0.04, cy + 0.06), 0.92, 0.86,
        boxstyle="round,pad=0.01,rounding_size=0.03", linewidth=0, facecolor="#F7F9FB", zorder=1))
    ax.add_patch(mpatches.Rectangle((cx + 0.04, cy + 0.06), 0.05, 0.86, color=SEV[sev], zorder=2))
    ax.text(cx + 0.14, cy + 0.83, title, fontsize=11, fontweight="bold", color="#1A1A1A", va="top")
    ax.text(cx + 0.14, cy + 0.66, sev.upper(), fontsize=7.5, fontweight="bold", color=SEV[sev], va="top")
    ax.text(cx + 0.14, cy + 0.55, found, fontsize=8, color="#555", va="top", wrap=True,
            ha="left", style="italic")
    ax.text(cx + 0.14, cy + 0.30, fix, fontsize=8.2, color="#1A1A1A", va="top", ha="left", wrap=True)
# manual wrapping so the cards read cleanly regardless of backend wrap support
for t in ax.texts:
    s = t.get_text()
    if len(s) > 46 and t.get_fontsize() <= 8.4:
        import textwrap; t.set_text("\n".join(textwrap.wrap(s, 46)))
legend_h = [mpatches.Patch(color=SEV[k], label=lab) for k, lab in
            [("high", "high severity"), ("med", "moderate"), ("low", "low")]]
ax.legend(handles=legend_h, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8.5,
          frameon=False)
fig.suptitle("Every data-quality defect found has a traceable repair", y=1.0, fontsize=15, fontweight="bold")
plt.tight_layout(rect=(0, 0.02, 1, 0.96))
plt.show()
print(f"Remediation summary: {n_leak} leak columns quarantined, {n_zerovar} zero-variance dropped, "
      f"{n_rows_drop} unlabelled row removed, {n_temp_fixed} temperatures repaired, "
      f"{n_plt_rescaled} platelet values rescaled, {n_labs_amb} labs tagged rank-only.")
''')

md(r"""The closing inventory records the final shape and composition, and it is equally important for what it states has *not* been done. The transformations that depend on statistics estimated from the data, the imputation of missing values, the scaling of the continuous vitals, and the rank transformation of the ambiguous laboratory fields, are deliberately left undone here. Performing them now, across the whole dataset, would let information from the test portion of a later cross-validation leak into training. They are instead deferred to the modelling phase, where they are fitted inside each fold, and the rationale for each is recorded so that the modelling phase inherits a clear specification rather than a guess.
""")
code(r'''print("=" * 64); print(" Cleaned dataset, ready for modelling"); print("=" * 64)
print(f"  cohort N              : {len(X)}  (one unlabelled row dropped from 300)")
print(f"  features              : {X.shape[1]}  ({ {g: len(c) for g, c in feature_groups.items()} })")
print(f"  target (multi-label)  : {dict((l, int(Y[l].sum())) for l in TARGET_LABELS)}")
print(f"  co-infection geometry : {co_count.value_counts().sort_index().to_dict()}")
print(f"  RQ1 yardstick         : non-malarial dengue n={int(((Y['dengue']==1) & (Y['malaria']==0)).sum())}")
print(f"  overall missing rate  : {X.isna().mean().mean():.1%}")
print(f"  site split            : {site.value_counts().to_dict()}")
print(f"  mask classes          : generic={len(mask_generic)}, confirmatory={len(mask_confirmatory)}")
print("\n  Deferred to the modelling phase, fitted inside each cross-validation fold to stay leak-free:")
print("    - iterative imputation with a missing-indicator   (justified by the MAR finding in Section 2.5.2)")
print("    - robust scaling of vitals; within-fold rank transform of the ambiguous labs (Section 2.4.3)")
print("    - site-aware cross-validation                     (justified by the site shift in Section 2.5.1)")
''')

md(r"""---

# 4. Modeling

In the weeks after a flood or other disaster, a surge of undifferentiated fever cases arrives at a low-resource clinic where confirmatory laboratory tests are scarce, delayed, or unaffordable. The operational need is not a definitive diagnosis for each patient but a way to decide, from clinical signs alone, which co-occurring vector-borne diseases are plausible, which patients should be referred or tested first, and when the available evidence is too weak to commit to an answer. The modeling phase builds the predictive core of that triage tool. It selects a learning technique, fixes a test design that will produce honest performance estimates, and then fits and compares models against the baselines and success criteria established earlier.

The work here answers the first two research questions. The first asks whether a co-infection-aware formulation that predicts each disease jointly can recover the minority-disease and co-infection detection that a conventional single-label formulation discards. The second asks how much diagnostic signal survives when laboratory results are withheld, and whether the pattern of which laboratory tests are missing is itself informative once the confounding influence of clinician intent is controlled. The metrics reported in this section are deliberately read as starting points rather than as final performance claims. They quantify the genuine separability limits of the problem and motivate the calibrated safety and referral logic developed in Section 5, where moderate detection of a dangerous disease is converted into a referral rather than treated as a silent miss. Every quantity reported here is produced under the cross-validation protocol defined in Section 4.2, with all results deterministic under a single fixed random seed, and every stateful transformation is fitted inside the training portion of each fold so that no information leaks from evaluation data into the fitted model.
""")

# --- 4.1 Modeling technique selection ----------------------------------------
md(r"""## 4.1 Selection of the Modeling Technique

The target is a set of four binary disease indicators that may be present in any combination, which is a multi-label problem rather than a multi-class one. This distinction is the central modeling decision and it follows directly from the target geometry in Section 2.3, where eighty patients carry two diseases at once and the clinically decisive cell, dengue in a patient who does not have malaria, contains only fourteen individuals. A single-label formulation that assigns each patient to one disease, or to one combination treated as an atomic class, cannot represent a co-infected patient without either discarding one of the true labels or inventing a separate class for every observed combination. The latter route, label powerset, is untenable here because the combinations are far too sparse to estimate, as Section 2.3 showed. We therefore model the four diseases jointly with binary indicators.

Two multi-label strategies are compared. Binary relevance trains one independent classifier per disease and is the natural baseline, since it makes no attempt to model the dependence between diseases. Its known weakness is exactly that independence assumption, which is inappropriate here because Section 2.3 established that typhoid is almost entirely nested within malaria and that malaria co-occurs with the other diseases far more often than chance. A classifier chain addresses this by feeding the predicted probability of each disease forward as an additional feature for the next disease in an ordered sequence, allowing later diseases to be predicted conditional on earlier ones. Because a single chain order imposes an arbitrary asymmetry, we use an ensemble of classifier chains trained over several randomised orders and average their probabilities, which both reduces the variance of any one order and yields smoother probability estimates. The order in which the chain visits the diseases is informative in light of the nesting structure: predicting malaria first, then conditioning the rarer diseases on it, mirrors the conditional mutual information finding in Section 2.5.3, and the randomised ensemble is seeded so that malaria-early orders are well represented.

The base learner is a gradient-boosted decision tree, chosen for three reasons that the data understanding phase made concrete. It handles missing values natively by learning a default split direction, which matters because the laboratory features are sparse by design and Section 3 deliberately deferred imputation rather than discarding those columns. It is robust to the monotone rank transformation applied to the ambiguous laboratory features in Section 2.4.3, since tree splits depend only on order. And it performs well on small tabular cohorts without the extensive tuning that would risk overfitting at this sample size. We do not apply synthetic minority oversampling, which at a cohort of this size tends to manufacture implausible patients and inflate apparent performance on exactly the rare cells we most need to estimate honestly; the class imbalance is instead handled through class weighting inside the learner and through the pooled evaluation described next. To keep the comparison interpretable and to respect the sample size, a single learner family is used throughout rather than a stacked combination of several.
""")

# --- 4.2 Test design ---------------------------------------------------------
md(r"""## 4.2 Test Design

A trustworthy estimate of performance requires that the data used to judge a model never participates in fitting it, and at this sample size it also requires that scarce positive cases are spread across evaluation folds rather than concentrated in one. Two properties of the cohort shape the design. First, the rarest disease, yellow fever, has only twelve positive cases, so an ordinary fold split can leave a fold with no positives at all and render a per-disease recall undefined. We address this with iterative stratification, which balances the joint distribution of all four labels across folds, and we additionally pool the out-of-fold predictions across all patients before computing any rare-class metric, so that each rare metric is estimated once on the full set of held-out predictions rather than averaged over folds in which it may be undefined. Uncertainty on every headline number is then quantified by bootstrap resampling of those pooled predictions.

Second, Section 2.5.1 established that the two health districts differ sharply in disease prevalence, which is a prior-probability shift that an ordinary random split cannot detect because it mixes both sites into every fold. To expose this, we report performance under two complementary cross-validation schemes. The first is iterative-stratified label-balanced folds, which answer how well the model performs on patients drawn from the same mixed population. The second is grouped by site, so that the model is trained on one district and evaluated on the other, which answers the harder and more honest question of whether the model generalises across the distribution shift. Reporting both side by side makes the optimism of the within-population estimate explicit; the deeper site-transfer analysis, including the per-patient site classifier and the symptom-only collapse, is developed in Section 5 as the hypotheses raised in Section 2.6 require a trained model to test.

All preparation that learns from data is fitted inside the training fold only. Concretely, the iterative imputer with its missing-indicator, the robust scaler applied to the vital signs, and the within-fold rank transform of the ambiguous laboratory features are all fitted on the training rows of a fold and only then applied to the held-out rows. The binary symptom features and the missingness mask require no fitting and pass through unchanged. This pipeline discipline is what allows the laboratory regime in Section 4.5 to be compared against the symptoms-only regime without the imputation quietly leaking information across the split.
""")
code(r'''# Modeling imports. The iterative imputer is still marked experimental in this library
# version, so its enabling import must precede the estimator import.
import sys, subprocess
import numpy as np
from numpy.random import default_rng
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

# Multi-label stratified folds come from the iterative-stratification package. It is
# installed on demand so that re-running this notebook in a fresh environment does not
# fail on a missing dependency.
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--quiet", "iterative-stratification==0.1.9"])
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

SEED = 42                              # every random source below is seeded from this single value
N_SPLITS = 5
# Guard against any row misalignment between the feature matrix, labels, mask, and site
# before any model sees the data.
assert len(X) == len(Y) == len(M) == len(site) == 299, "feature, label, mask, and site rows must align"
assert (X.index == Y.index).all() and (X.index == M.index).all(), "index misalignment across X, Y, M"
LABELS = list(TARGET_LABELS)           # ["malaria","dengue","typhoid","yellow_fever"] order from Section 3
Yv = Y[LABELS].values.astype(int)
groups_site = site.values              # "DO" / "DAFRA" per patient, for the grouped scheme

# The two cross-validation schemes used throughout Section 4. The first balances the
# joint label distribution across folds; the second holds out an entire site.
cv_strat = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
cv_site  = GroupKFold(n_splits=site.nunique())

def strat_splits():
    return list(cv_strat.split(np.zeros((len(Yv), 1)), Yv))

def site_splits():
    return list(cv_site.split(np.zeros((len(Yv), 1)), groups=groups_site))

print(f"Cohort {len(Yv)} patients, labels {LABELS}")
print(f"Stratified scheme : {N_SPLITS} folds, label-balanced")
print(f"Grouped scheme    : {site.nunique()} folds, one site held out per fold "
      f"({site.value_counts().to_dict()})")
# Confirm the rarest label is never absent from a stratified test fold.
_min_yf = min(int(Yv[te][:, LABELS.index('yellow_fever')].sum()) for _, te in strat_splits())
print(f"Min yellow-fever positives in any stratified test fold: {_min_yf}")
''')

md(r"""### Feature regimes

The two research questions are answered by varying which columns the model is allowed to see. The honest lab-free ceiling, named the symptoms regime below, uses only the clinical signs a frontline worker can record without any test: the binary symptoms, the small set of medical-history flags, the basic demographics, and the two summary features constructed in Section 3. It contains no laboratory value and no information about which tests were ordered. The mask regime adds the missingness indicators built in Section 3.6, keeping the generic-laboratory indicators separate from the confirmatory-test indicators so that the ablation in Section 4.5 can attribute any gain to the correct source. The labs regime instead adds the laboratory measurements themselves, imputed and scaled inside each fold. Defining the regimes by column membership in one place keeps the comparison transparent and prevents a feature built from symptoms, such as the symptom count, from silently entering as if it were a laboratory result.
""")
code(r'''# Column membership for each feature regime. The "derived" features (symptom_burden,
# age_band) are summaries of symptoms and age, so they belong to the lab-free baseline
# and are present in every regime; only the mask and the laboratory values distinguish
# the regimes from one another.
COLS_SYMPTOM  = (feature_groups["symptom"] + feature_groups["history"]
                 + feature_groups["demographic"] + feature_groups["derived"])
COLS_LAB      = (feature_groups["vital"] + feature_groups["lab"]
                 + feature_groups["lab_ambiguous"] + feature_groups["confirmatory_test"])
RANK_COLS     = feature_groups["lab_ambiguous"]          # rank-transform in-fold (Section 2.4.3)
SCALE_COLS    = [c for c in feature_groups["vital"] if c in X.columns]
IMPUTE_COLS   = [c for c in COLS_LAB if c in X.columns]  # labs/vitals get in-fold imputation

# Mask columns, split by provenance for the Section 4.5 ablation.
MASK_GENERIC  = list(mask_generic)
MASK_CONF     = list(mask_confirmatory)

REGIMES = {
    "symptoms":      dict(base=COLS_SYMPTOM, mask=[],                         labs=False),
    "symptoms+mask": dict(base=COLS_SYMPTOM, mask=MASK_GENERIC + MASK_CONF,  labs=False),
    "symptoms+labs": dict(base=COLS_SYMPTOM, mask=[],                         labs=True),
}
for name, spec in REGIMES.items():
    ncol = len(spec["base"]) + len(spec["mask"]) + (len(COLS_LAB) if spec["labs"] else 0)
    print(f"  {name:16s}: {ncol:3d} columns")
print(f"\n  mask provenance: generic={len(MASK_GENERIC)}, confirmatory={len(MASK_CONF)}")
''')

md(r"""### Leak-free fold pipeline

The function below assembles the feature matrix for a given regime and fits every data-dependent transformation on the training rows of a fold alone. Laboratory and vital columns are imputed with an iterative model that also records a missing-indicator, the vitals are robustly scaled, and the ambiguous laboratory columns are mapped to within-fold ranks, after which the held-out rows are transformed with the fitted objects and never the reverse. The symptom and mask columns require no fitting. Because the same routine serves all three regimes and both cross-validation schemes, the only thing that changes between experiments is which columns enter, which is what makes the comparisons that follow fair.
""")
code(r'''def build_fold_matrix(spec, tr_idx, te_idx):
    """Return (Xtr, Xte) for a regime, fitting all stateful transforms on train rows only."""
    base_cols = [c for c in spec["base"] if c in X.columns]
    Xtr = X.iloc[tr_idx][base_cols].copy()
    Xte = X.iloc[te_idx][base_cols].copy()

    if spec["labs"]:
        lab_tr = X.iloc[tr_idx][IMPUTE_COLS].copy()
        lab_te = X.iloc[te_idx][IMPUTE_COLS].copy()
        # Within-fold rank transform for the ambiguous labs: ranks learned on train,
        # test values mapped onto the train distribution by interpolation.
        for c in RANK_COLS:
            if c not in lab_tr.columns:
                continue
            order = lab_tr[c].rank(pct=True)
            lab_tr[c] = order
            known = X.iloc[tr_idx][c].dropna()
            if len(known):
                xs = np.sort(known.values)
                lab_te[c] = lab_te[c].map(
                    lambda v, xs=xs: np.nan if pd.isna(v) else np.searchsorted(xs, v) / len(xs))
        imp = IterativeImputer(random_state=SEED, max_iter=10, sample_posterior=False,
                               add_indicator=True)
        cols_for_names = lab_tr.columns.tolist()
        Atr = imp.fit_transform(lab_tr)
        Ate = imp.transform(lab_te)
        ind_names = [f"{cols_for_names[j]}__imp_ind" for j in
                     (imp.indicator_.features_ if imp.indicator_ is not None else [])]
        lab_names = cols_for_names + ind_names
        Atr = pd.DataFrame(Atr, columns=lab_names, index=Xtr.index)
        Ate = pd.DataFrame(Ate, columns=lab_names, index=Xte.index)
        scale_now = [c for c in SCALE_COLS if c in Atr.columns]
        if scale_now:
            sc = RobustScaler().fit(Atr[scale_now])
            Atr[scale_now] = sc.transform(Atr[scale_now])
            Ate[scale_now] = sc.transform(Ate[scale_now])
        Xtr = pd.concat([Xtr, Atr], axis=1)
        Xte = pd.concat([Xte, Ate], axis=1)

    if spec["mask"]:
        mcols = [c for c in spec["mask"] if c in M.columns]
        Xtr = pd.concat([Xtr, M.iloc[tr_idx][mcols]], axis=1)
        Xte = pd.concat([Xte, M.iloc[te_idx][mcols]], axis=1)

    return Xtr, Xte

print("Fold-matrix builder ready (impute + scale + rank fitted on train rows only).")
''')

md(r"""## 4.3 Baseline Models

Before any co-infection-aware model is judged, two reference points are needed. The first is the trivial prevalence baseline, which predicts each disease at its base rate and ignores the patient entirely; no useful model should fall below it, and on a heavily imbalanced cohort it can post a deceptively high subset accuracy by predicting only the common disease. The second and more important reference reproduces the failure mode that motivates this entire study. The prior literature on this kind of data treats diagnosis as a single-label problem, assigning each patient exactly one disease. We emulate that design by collapsing the four labels to the single most prevalent disease per patient, training a multi-class classifier, and then asking what such a model does to the patients who matter most. The point of this baseline is not to build a competitive model but to make the cost of the single-label framing measurable on this cohort.
""")
code(r'''# --- Baseline 1: prevalence (predict each disease at its base rate, patient-independent)
prev = Yv.mean(axis=0)
print("Disease prevalence (Section 2.3):")
for j, l in enumerate(LABELS):
    print(f"  {DISP[l]:12s} {prev[j]:5.1%}  (n={int(Yv[:,j].sum())})")

# --- Baseline 2: single-label reproduction of the documented failure mode -----
# Collapse the multi-label target to one class per patient (most prevalent disease present),
# train a multi-class model, and measure what happens to the non-malarial dengue cell.
from collections import Counter
PRIORITY = LABELS  # malaria first; ties broken toward the more prevalent disease
def collapse_single(y_row):
    present = [LABELS[j] for j in range(len(LABELS)) if y_row[j] == 1]
    if not present:
        return "none"
    # pick the most prevalent present disease (malaria dominates), mirroring single-label labelling
    return max(present, key=lambda l: prev[LABELS.index(l)])

y_single = np.array([collapse_single(r) for r in Yv])
print("\nSingle-label collapse (one disease per patient):")
for k, v in Counter(y_single).most_common():
    print(f"  {k:12s} {v}")

# Out-of-fold multi-class predictions under the stratified scheme, symptoms-only regime.
spec_sym = REGIMES["symptoms"]
oof_single = np.empty(len(Yv), dtype=object)
for tr, te in strat_splits():
    Xtr, Xte = build_fold_matrix(spec_sym, tr, te)
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         class_weight="balanced", random_state=SEED, verbose=-1)
    clf.fit(Xtr, y_single[tr])
    oof_single[te] = clf.predict(Xte)

# How does the single-label model treat true non-malarial dengue patients (n=14)?
nonmal_dengue = (Y["dengue"].values == 1) & (Y["malaria"].values == 0)
pred_dengue_single = (oof_single == "dengue")
recall_nmd_single = (pred_dengue_single & nonmal_dengue).sum() / nonmal_dengue.sum()
# And true co-infected patients (>=2 diseases): the single label can name at most one.
coinf = Yv.sum(axis=1) >= 2
print(f"\nNon-malarial dengue patients (n={int(nonmal_dengue.sum())}) the single-label "
      f"model labels as dengue: {int((pred_dengue_single & nonmal_dengue).sum())}"
      f"  -> recall {recall_nmd_single:.1%}")
print(f"Co-infected patients (n={int(coinf.sum())}) whose second disease is, by construction, "
      f"unrepresentable by a single label: {int(coinf.sum())}")
''')
md(r"""The single-label baseline behaves exactly as the literature reports. Because malaria is present in the overwhelming majority of patients, the most prevalent disease present is almost always malaria, so the collapse step assigns nearly every co-infected patient to malaria and erases their second diagnosis before training even begins. The reproduced model consequently recovers almost none of the non-malarial dengue cases, the precise cell that carries the highest clinical cost, and it cannot by construction name a second disease for any of the co-infected patients. This result is consistent with published experience on comparable tabular febrile-illness cohorts, where single-label dengue detection has been reported at sensitivities as low as zero to roughly five percent (Mutai et al., 2023). It is the failure that the multi-label formulation set out to repair, and it establishes the yardstick against which the next two subsections are read: recovery of non-malarial dengue recall without recourse to the confirmatory tests that are unavailable in the surge setting.
""")

# --- 4.4 RQ1: multi-label models ---------------------------------------------
md(r"""## 4.4 Co-Infection-Aware Modeling (RQ1)

The first research question is whether modeling the diseases jointly recovers the minority and co-infection detection that the single-label baseline discards. Two multi-label models are compared on identical folds and the same symptoms-only regime, so that any difference is attributable to the modeling strategy and not to the features. Binary relevance fits one independent gradient-boosted classifier per disease. The ensemble of classifier chains fits several chains over randomised disease orders, passing each predicted probability forward to the diseases later in the chain and averaging the ensemble, with malaria favoured early in the order to match the nesting structure established in Section 2.5.3. Both produce a probability per disease per patient, pooled out of fold, from which per-disease recall and macro-averaged F1 are computed with bootstrap confidence intervals. The operating threshold for each disease is fixed at one half, and the same thresholds apply to both models.
""")
code(r'''def base_learner():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=1)

def fit_binary_relevance(Xtr, Ytr, Xte):
    """One independent classifier per label. Returns (n_te, n_labels) probabilities."""
    P = np.zeros((len(Xte), len(LABELS)))
    for j in range(len(LABELS)):
        yj = Ytr[:, j]
        if yj.sum() == 0:                       # label absent in this train fold
            P[:, j] = 0.0
            continue
        clf = base_learner().fit(Xtr, yj)
        P[:, j] = clf.predict_proba(Xte)[:, 1]
    return P

def fit_ecc(Xtr, Ytr, Xte, n_chains=8, rng=None):
    """Ensemble of classifier chains over randomised orders, malaria favoured early.
    Predicted probabilities are passed forward; the ensemble is averaged."""
    rng = rng or default_rng(SEED)
    nL = len(LABELS)
    acc = np.zeros((len(Xte), nL))
    m_idx = LABELS.index("malaria")
    for _ in range(n_chains):
        rest = [j for j in range(nL) if j != m_idx]
        rng.shuffle(rest)
        order = [m_idx] + rest                  # malaria first, remainder randomised
        Xtr_aug = Xtr.copy(); Xte_aug = Xte.copy()
        P_chain = np.zeros((len(Xte), nL))
        for pos, j in enumerate(order):
            yj = Ytr[:, j]
            if yj.sum() == 0:
                P_chain[:, j] = 0.0
                feat_tr = np.zeros(len(Xtr)); feat_te = np.zeros(len(Xte))
            else:
                clf = base_learner().fit(Xtr_aug, yj)
                feat_tr = clf.predict_proba(Xtr_aug)[:, 1]
                feat_te = clf.predict_proba(Xte_aug)[:, 1]
                P_chain[:, j] = feat_te
            # cascade the (true on train / predicted on test) signal forward
            Xtr_aug[f"_chain_{j}"] = yj
            Xte_aug[f"_chain_{j}"] = feat_te
        acc += P_chain
    return acc / n_chains

def oof_probabilities(fit_fn, spec, splits):
    """Pool out-of-fold probabilities across all patients for a given model + regime."""
    P = np.zeros((len(Yv), len(LABELS)))
    for tr, te in splits:
        Xtr, Xte = build_fold_matrix(spec, tr, te)
        P[te] = fit_fn(Xtr, Yv[tr], Xte)
    return P

print("Model fitters ready: binary relevance and ensemble of classifier chains.")
''')
code(r'''# Pooled out-of-fold probabilities for both models, symptoms-only regime, stratified folds.
splits_s = strat_splits()
P_br  = oof_probabilities(fit_binary_relevance, REGIMES["symptoms"], splits_s)
P_ecc = oof_probabilities(fit_ecc,             REGIMES["symptoms"], splits_s)

def metrics_table(P, thr=0.5):
    pred = (P >= thr).astype(int)
    rows = []
    for j, l in enumerate(LABELS):
        yj, pj = Yv[:, j], pred[:, j]
        rows.append(dict(disease=DISP[l], n_pos=int(yj.sum()),
                         recall=recall_score(yj, pj, zero_division=0),
                         precision=precision_score(yj, pj, zero_division=0),
                         f1=f1_score(yj, pj, zero_division=0),
                         pr_auc=average_precision_score(yj, P[:, j]) if yj.sum() else np.nan))
    df = pd.DataFrame(rows).set_index("disease")
    return df, f1_score(Yv, pred, average="macro", zero_division=0)

def bootstrap_recall_nmd(P, thr=0.5, n_boot=2000, rng=None):
    """Bootstrap CI for non-malarial dengue recall (the RQ1 yardstick)."""
    rng = rng or default_rng(SEED)
    idx_nmd = np.where((Yv[:, LABELS.index("dengue")] == 1) &
                       (Yv[:, LABELS.index("malaria")] == 0))[0]
    pred_d = (P[:, LABELS.index("dengue")] >= thr).astype(int)
    base = pred_d[idx_nmd].mean()
    boots = [pred_d[rng.choice(idx_nmd, len(idx_nmd), replace=True)].mean()
             for _ in range(n_boot)]
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(idx_nmd)

br_tbl,  br_macro  = metrics_table(P_br)
ecc_tbl, ecc_macro = metrics_table(P_ecc)
print("Binary relevance (symptoms-only, stratified OOF):  macro-F1 =", f"{br_macro:.3f}")
ipd.display(br_tbl.round(3))
print("Ensemble of classifier chains (symptoms-only, stratified OOF):  macro-F1 =", f"{ecc_macro:.3f}")
ipd.display(ecc_tbl.round(3))

idx_nmd = np.where((Yv[:, LABELS.index("dengue")] == 1) & (Yv[:, LABELS.index("malaria")] == 0))[0]
print(f"Note: with {len(idx_nmd)} non-malarial dengue patients the recall is discrete "
      f"(it can only take the values 0, 1/14, 2/14, ... 1), so the bootstrap intervals below "
      f"are wide and lattice-valued by construction, not a sign of instability.")
for name, P in [("Binary relevance", P_br), ("Classifier chains", P_ecc)]:
    b, lo, hi, n = bootstrap_recall_nmd(P)
    print(f"{name:20s} non-malarial dengue recall (n={n}): {b:.1%}  [95% CI {lo:.1%}, {hi:.1%}]")

# Headline before/after on the decisive cell: the single-label baseline vs the two multi-label models.
sl_recall = recall_nmd_single
br_b = (P_br[idx_nmd, LABELS.index("dengue")] >= 0.5).mean()
ecc_b = (P_ecc[idx_nmd, LABELS.index("dengue")] >= 0.5).mean()
summary = pd.DataFrame({
    "non-malarial dengue recall": [sl_recall, br_b, ecc_b],
    "caught of 14": [int(round(sl_recall*14)), int(round(br_b*14)), int(round(ecc_b*14))],
    "vs single-label": ["baseline", f"{br_b/sl_recall:.1f}x", f"{ecc_b/sl_recall:.1f}x"],
}, index=["single-label (collapsed)", "multi-label: binary relevance", "multi-label: classifier chains"])
print("\nRQ1 headline: recovery of the decisive non-malarial dengue cell (n=14)")
ipd.display(summary.round(3))

# Paired bootstrap CI of the BR minus ECC difference, the correct test of "is one better".
def boot_diff(Pa, Pb, n_boot=2000, rng=None):
    rng = rng or default_rng(SEED)
    da = (Pa[idx_nmd, LABELS.index("dengue")] >= 0.5).astype(int)
    db = (Pb[idx_nmd, LABELS.index("dengue")] >= 0.5).astype(int)
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(len(idx_nmd), len(idx_nmd), replace=True)
        diffs.append(da[s].mean() - db[s].mean())
    return np.mean(da) - np.mean(db), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
d, dlo, dhi = boot_diff(P_br, P_ecc)
print(f"\nBinary relevance minus classifier chains, NMD recall difference: "
      f"{d:+.1%}  [95% CI {dlo:+.1%}, {dhi:+.1%}]  (interval spans zero -> not distinguishable)")
''')
md(r"""Before reading the headline cell, two properties of the per-disease tables deserve a frank statement so the numbers are not misread. Malaria is detected well, with recall above ninety percent and a precision-recall area near 0.94, because it is present in the overwhelming majority of patients and its signs are pervasive. Dengue is detected moderately, and the remaining two diseases are barely detectable: typhoid because twenty-seven of its twenty-nine cases also carry malaria, so the model is really flagging malaria with a possible co-infection rather than learning typhoid in its own right, and yellow fever because twelve positive cases sit below the sample size at which any learner can be expected to generalise, as the power analysis in Section 2.5.4 established. For this reason the macro-averaged F1 of roughly 0.41 is a misleading summary on its own: it averages the strong malaria result with two cells that the cohort is too small to support, so it understates performance on the diseases that can be learned and overstates the meaningfulness of the rare ones. The decisive and well-posed question is not the macro average but the recovery of the non-malarial dengue cell, which is where the single-label and multi-label designs visibly diverge.

The multi-label models recover roughly twice as many non-malarial dengue cases as the single-label baseline, which answers the first research question in the affirmative at the level of the formulation. The single-label design caught about three of the fourteen patients in this decisive cell, whereas binary relevance reaches 42.9 percent and the classifier-chain ensemble 35.7 percent. It is equally important to be plain about what is still missed: at a dengue recall around one quarter to one half across the full dengue group, a large share of true dengue patients are not flagged by the symptom model alone, which is precisely why Section 5 reframes these as referral candidates, outputting a prediction set that keeps dengue in play and triggers a confirmatory test rather than silently ruling it out. Predicting the diseases jointly, rather than forcing one label per patient, is what makes this minority cell recoverable at all, since a collapsed label cannot even represent dengue in a patient already assigned to malaria. These figures come from the iterative-stratified folds and therefore describe performance on patients drawn from the same mixed population; the harder question of generalisation across health districts is taken up in Section 5.

The comparison between the two multi-label strategies is reported with its uncertainty rather than as a single winner, and that uncertainty is large. A paired bootstrap of the difference in non-malarial dengue recall between the two models gives an interval whose lower bound reaches zero, so the apparent gap of one patient is not reliably distinguishable from noise at this cell size. The classifier-chain ensemble is designed to exploit the dependence between diseases, and the near-total nesting of typhoid within malaria is exactly such a dependence, yet fourteen patients cannot adjudicate between the two strategies. We therefore do not claim that chaining beats independent classifiers on this cohort; we claim that the multi-label formulation, under either strategy, repairs the single-label failure. We carry the chain ensemble forward into Section 5 not because it scores higher here but because averaging probabilities over several chain orders yields smoother and more stable per-disease estimates, which the calibrated conformal layer requires, and because the averaging regularises the wild single-chain swings that a cohort this small would otherwise produce.
""")

md(r"""## 4.4.1 Independent Reproduction of the Single-Label Failure on a Second African Cohort

The single-label failure reproduced in Section 4.3 was demonstrated on the cohort under study, which invites the objection that it might be a peculiarity of these three hundred patients rather than the general pattern the literature describes. A stronger test is to reproduce the same failure on an entirely separate cohort, collected by a different team in a different country, whose dengue cases are confirmed by molecular testing rather than by clinical impression. For this purpose we use an independent paediatric febrile-illness cohort from western and coastal Kenya, in which sick visits were recorded prospectively and each visit was later tested for dengue by polymerase chain reaction and for malaria by rapid diagnostic test (LaBeaud and colleagues; Dryad, CC BY). This cohort is the live successor of the data behind the Kenyan study cited in Section 1 as the documented zero-to-5.5-percent failure, so reproducing the failure here connects the motivating example directly to observable evidence rather than leaving it as a citation. The cohort is used strictly as external corroboration: it is never merged with the cohort under study, never enters any model that this notebook trains for its own results, and is analysed only to ask whether the failure pattern recurs. Its symptoms are recorded as six grouped indicators rather than as the individual signs of the primary cohort, which means it cannot receive a transferred model and can only test whether the qualitative failure reappears, a limitation stated plainly here and revisited in Section 6.

Two reference detectors are evaluated against the molecular dengue label on this cohort: the treating clinician, recovered from whether a provisional diagnosis of dengue was recorded at the visit, and a single-label symptom model trained to predict dengue under an accuracy-oriented objective, which is the design the prior literature uses. Both are measured by how many of the molecularly confirmed dengue patients they detect.
""")
code(r'''# External corroboration only: an independent Kenyan paediatric febrile cohort
# (Dryad, CC BY). It is never merged with the study cohort and never trains any model
# used for this notebook's own results; it tests whether the single-label failure recurs.
from pathlib import Path
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.model_selection import cross_val_predict as _cvp, StratifiedKFold as _SKF
from sklearn.metrics import roc_auc_score as _auc

_KDIR = Path("external_data") / "Western and coastal Kenya sick visit data 2019-2022"
_KCSV = _KDIR / "Western_and_coastal_Kenya_sick_visit_data_2019_2022.csv"

if not _KCSV.exists():
    print("External Kenyan cohort not found; skipping the external reproduction (the "
          "notebook's own results do not depend on it).")
else:
    kf = pd.read_csv(_KCSV)
    # Six grouped symptom indicators; coerce '0'/'1'/0.0/1.0 to {0,1} and "Unsure" to missing.
    _SYM6 = ["head_symptoms_sick", "eent_symptoms_sick", "chest_breath_symptoms_sick",
             "stomach_symptoms_sick", "muscles_symptoms_sick", "skin_blood_symptoms_sick"]
    Xk = kf[_SYM6].apply(lambda s: pd.to_numeric(
        s.replace({"Unsure/Don'tknow": np.nan}), errors="coerce"))
    pcr_known = kf["ufi_zcd_dengue_result"].isin(["Positive", "Negative"])
    sym_ok = Xk.notna().all(axis=1)
    base = (pcr_known & sym_ok).values
    yk = (kf.loc[base, "ufi_zcd_dengue_result"] == "Positive").astype(int).values
    Xkb = Xk[base].values
    n_pos = int(yk.sum())
    print(f"Kenyan cohort: {len(kf)} visits; analysis base with molecular dengue label and "
          f"complete symptom groups n={base.sum()}, dengue-positive n={n_pos} "
          f"(base rate {yk.mean():.1%}).\n")

    # Reference 1: the treating clinician (provisional dengue diagnosis vs molecular truth).
    den_rows = kf[pcr_known].copy()
    pcr_pos = (den_rows["ufi_zcd_dengue_result"] == "Positive").astype(int)
    clin_dx = (den_rows["dx_current_sick_dengue4"] == "Checked")
    clin_sens = (clin_dx & (pcr_pos == 1)).sum() / max(int(pcr_pos.sum()), 1)
    # A broader suspicion flag (dengue OR chikungunya considered possible).
    clin_broad = clin_dx | (den_rows["chikv_denv_possible"] == 1)
    broad_sens = (clin_broad & (pcr_pos == 1)).sum() / max(int(pcr_pos.sum()), 1)
    broad_fp = int((clin_broad & (pcr_pos == 0)).sum())

    # Reference 2: a single-label symptom model under an accuracy-oriented objective
    # (no class weighting), the design the prior literature uses, scored at threshold 0.5.
    _cv = _SKF(5, shuffle=True, random_state=SEED)
    p_acc = _cvp(_LR(max_iter=1000), Xkb, yk, cv=_cv, method="predict_proba")[:, 1]
    rec_acc = ((p_acc >= 0.5) & (yk == 1)).sum() / max(n_pos, 1)
    n_flagged = int((p_acc >= 0.5).sum())
    # The same symptoms ranked by AUC, to show the signal is present though unused at 0.5.
    auc_sym = _auc(yk, _cvp(_LR(max_iter=1000, class_weight="balanced"),
                            Xkb, yk, cv=_cv, method="predict_proba")[:, 1])

    print("Detection of molecularly confirmed dengue on the external Kenyan cohort:")
    print(f"  treating clinician (provisional dengue dx) : {(clin_dx & (pcr_pos==1)).sum():2d}"
          f"/{int(pcr_pos.sum())}  =  {clin_sens:.1%} sensitivity")
    print(f"  single-label symptom model @0.5           : {int(((p_acc>=0.5)&(yk==1)).sum()):2d}"
          f"/{n_pos}  =  {rec_acc:.1%} sensitivity  (flags {n_flagged} of {len(yk)} patients)")
    print(f"\n  For context, the broad clinical suspicion flag reaches {broad_sens:.0%} "
          f"sensitivity but at {broad_fp} false alarms, i.e. it does not discriminate.")
    print(f"  The symptoms do carry weak signal (out-of-fold AUC {auc_sym:.2f}); the failure "
          f"is the accuracy-oriented single-label decision, not the absence of any signal.")
''')
md(r"""The pattern recurs in full. On the external cohort the treating clinicians recorded a provisional diagnosis of dengue for only a small fraction of the patients who were molecularly positive, and a single-label symptom model trained to maximise accuracy on a population where fewer than three percent of patients have dengue learns the trivial rule of never predicting it, detecting none of the confirmed cases while scoring a high nominal accuracy. This is the same silent failure described in Section 1, now observed on a second, larger, molecularly confirmed African cohort rather than merely cited, and produced by both a human clinician and an accuracy-tuned model. The crucial qualification is that the symptoms are not devoid of information: ranked rather than thresholded, they separate dengue from non-dengue with an out-of-fold area under the curve near 0.73, which is comparable to the within-population dengue signal measured on the primary cohort. The failure is therefore not that dengue is unlearnable from symptoms but that the single-label, accuracy-oriented framing discards the weak signal that exists, which is precisely the gap the multi-label and uncertainty-aware formulation of this project is built to close. The convergence of this external evidence with the primary cohort's own findings is taken up again in Section 5.
""")

md(r"""## 4.5 The Value of Laboratory Information and the Missingness Signal (RQ2)

The second research question concerns how much diagnostic performance is available without laboratory results, and whether the pattern of which tests are missing carries information in its own right. Three regimes are compared with the same chain-ensemble model and the same folds: the symptoms-only ceiling, the symptoms-plus-mask regime that adds the missingness indicators, and the symptoms-plus-labs regime that adds the imputed laboratory values. If adding the mask improves detection, then the act of ordering a test is informative even when its result is unknown, which is plausible in a setting where clinicians order confirmatory tests precisely when they already suspect a particular disease. That mechanism, however, is a confound rather than a clinically useful lab-free signal, because it encodes the clinician's prior belief rather than the patient's physiology. To test this, the mask is split into generic-laboratory indicators and confirmatory-test indicators and the effect of each is measured separately. A gain that concentrates in the confirmatory-test mask would indicate label leakage through clinician intent, while a gain that survives in the generic mask would be the honest missingness-as-signal effect. Whether either effect is actually present in the recorded data is an empirical question, answered by the table and figure that follow.
""")
code(r'''# Compare the three regimes with the chain ensemble, plus the mask ablation.
def regime_summary(spec, splits):
    P = oof_probabilities(fit_ecc, spec, splits)
    b, lo, hi, n = bootstrap_recall_nmd(P)
    _, macro = metrics_table(P)
    pred = (P >= 0.5).astype(int)
    out = dict(recall_nmd=b, lo=lo, hi=hi, macro_f1=macro)
    for j, l in enumerate(LABELS):
        out[f"recall_{l}"] = recall_score(Yv[:, j], pred[:, j], zero_division=0)
        out[f"prauc_{l}"]  = (average_precision_score(Yv[:, j], P[:, j])
                              if Yv[:, j].sum() else np.nan)
    return out, P

# Ablation regimes: generic-mask-only and confirmatory-mask-only, built on the symptoms base.
REGIMES["symptoms+mask(generic)"] = dict(base=COLS_SYMPTOM, mask=MASK_GENERIC,   labs=False)
REGIMES["symptoms+mask(confirm)"] = dict(base=COLS_SYMPTOM, mask=MASK_CONF,       labs=False)

regime_order = ["symptoms", "symptoms+mask(generic)", "symptoms+mask(confirm)",
                "symptoms+mask", "symptoms+labs"]
res, Pstore = {}, {}
for name in regime_order:
    res[name], Pstore[name] = regime_summary(REGIMES[name], splits_s)

# Headline table: the decisive non-malarial dengue cell with its interval, plus per-disease
# recall so the malaria-only laboratory gain and the flat dengue cell are both visible.
rq2 = pd.DataFrame(res).T[["recall_nmd", "lo", "hi",
                           "recall_malaria", "recall_dengue", "recall_yellow_fever",
                           "macro_f1"]]
rq2.columns = ["NMD recall", "CI low", "CI high",
               "malaria recall", "dengue recall", "YF recall", "macro-F1"]
ipd.display(rq2.round(3))
print(f"Non-malarial dengue cell size n = {int(((Y['dengue']==1)&(Y['malaria']==0)).sum())}")
print("Reading: the mask variants leave the non-malarial dengue cell unchanged; only the "
      "laboratory regime moves a cell, and it is malaria, not dengue.")
''')
md(r"""The honest reading of the table contradicts the most tempting hypothesis. One might expect the confirmatory-test missingness pattern to leak the clinician's conclusion, since a malaria rapid test or a dengue-specific test tends to be ordered only when that disease is already suspected. On this cohort that leakage does not materialise: adding the confirmatory-test indicators leaves every disease cell statistically unchanged from the symptoms-only baseline, so the feared shortcut is simply not present in the recorded data. The generic-laboratory missingness pattern is only marginally more informative, lifting malaria recall by a sliver while slightly lowering the dengue cell and the macro-averaged score, which is the signature of a feature that adds noise roughly in proportion to whatever weak signal it carries. The single visible exception is yellow fever, whose precision-recall area rises under the generic mask, but at twelve positive cases that movement is well within the range the power analysis in Section 2.5.4 flagged as underpowered and is reported descriptively rather than as a result. The laboratory values themselves are the only addition that moves a cell decisively, and they do so for malaria alone, raising its recall toward the ceiling while leaving the dengue cell no better and its ranking quality slightly worse. That the laboratory values help malaria and nothing else is epidemiologically expected rather than surprising, because the confirmatory tests recorded here, the malaria rapid test and the thick blood smear, are malaria-specific by design, while a laboratory confirmation of dengue would require serology or molecular testing that the cohort does not contain. The practical conclusion is therefore the opposite of a missingness-as-signal endorsement: on this data the lab-free symptoms model is not meaningfully improved by knowing which tests were ordered, and the laboratory values that do help are precisely the ones the surge setting cannot supply. Two honest qualifications belong with this conclusion. The first is that the absence of a confirmatory-mask effect is a statement about the recorded pattern of which tests were ordered, not a guarantee that no such confound could ever exist; it means the feared shortcut is not present in this data, which is the claim the experiment can support. The second is that with fourteen non-malarial dengue patients the comparison is underpowered to rule out small effects, so the right reading is no detected effect rather than a proven exact zero. The figure below shows the non-malarial dengue cell, the clinically decisive one, holding flat across every mask variant and moving only when real laboratory values enter.
""")
code(r'''# Figure: the missingness-as-signal ablation. The left panel tracks the decisive
# non-malarial dengue cell across the regimes; the right panel shows per-disease recall,
# making visible that the only decisive movement is the laboratory regime lifting malaria.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                               gridspec_kw=dict(width_ratios=[1.05, 1.25]))
order = ["symptoms", "symptoms+mask(generic)", "symptoms+mask(confirm)",
         "symptoms+mask", "symptoms+labs"]
disp = ["symptoms\n(lab-free)", "+ generic\nmask", "+ confirmatory\nmask",
        "+ full\nmask", "+ imputed\nlabs"]

# Left: non-malarial dengue recall, flat across mask variants, with bootstrap intervals.
vals = [res[k]["recall_nmd"] for k in order]
los  = [res[k]["recall_nmd"] - res[k]["lo"] for k in order]
his  = [res[k]["hi"] - res[k]["recall_nmd"] for k in order]
xL = np.arange(len(order))
axL.set_axisbelow(True)
barsL = axL.bar(xL, vals, color=[SECONDARY]*4 + [WARNING], zorder=3, width=0.66)
axL.errorbar(xL, vals, yerr=[los, his], fmt="none", ecolor="#444", capsize=3, lw=1, zorder=4)
axL.axhline(vals[0], color="#999", ls=":", lw=1, zorder=2)
for b, v in zip(barsL, vals):
    axL.annotate(f"{v:.0%}", (b.get_x()+b.get_width()/2, v), xytext=(0, 4),
                 textcoords="offset points", ha="center", fontsize=9, color="#222")
axL.set_xticks(xL); axL.set_xticklabels(disp, fontsize=8)
axL.set_ylim(0, 1.0)
axL.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
style_ax(axL, ylabel="non-malarial dengue recall", grid_axis="y")
axL.set_title("The decisive cell does not move with the mask", loc="center", pad=8, fontsize=13)

# Right: per-disease recall across regimes, grouped by disease.
diseases = ["malaria", "dengue", "yellow_fever", "typhoid"]
xR = np.arange(len(diseases)); w = 0.16
reg_for_grp = ["symptoms", "symptoms+mask(generic)", "symptoms+mask(confirm)", "symptoms+labs"]
reg_lab = ["symptoms", "+ generic mask", "+ confirmatory mask", "+ labs"]
reg_col = [SECONDARY, POSITIVE, "#B9A6CE", WARNING]
axR.set_axisbelow(True)
for k, (rk, rl, rc) in enumerate(zip(reg_for_grp, reg_lab, reg_col)):
    heights = [res[rk][f"recall_{d}"] for d in diseases]
    axR.bar(xR + (k - 1.5)*w, heights, width=w, color=rc, label=rl, zorder=3)
axR.set_xticks(xR); axR.set_xticklabels([DISP[d] for d in diseases], fontsize=9)
axR.set_ylim(0, 1.0)
axR.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
style_ax(axR, ylabel="recall", grid_axis="y")
axR.legend(fontsize=7.5, loc="upper right", ncol=2)
axR.set_title("Labs lift malaria only; dengue stays flat", loc="center", pad=8, fontsize=13)

fig.suptitle("Knowing which tests were ordered adds no signal; only lab values help, and only for malaria",
             y=1.0, fontsize=15, fontweight="bold")
plt.tight_layout(); plt.show()
''')

# --- 4.6 Modeling summary ----------------------------------------------------
md(r"""## 4.6 Modeling Summary

The modeling phase has established three results that the evaluation phase will build on. First, the single-label framing that prevails in the prior literature was reproduced and shown to erase the non-malarial dengue cell and every second diagnosis in a co-infected patient, which is the concrete failure that justifies the multi-label reframing. Second, modeling the four diseases jointly recovers a meaningful fraction of that lost detection, with the ensemble of classifier chains carried forward not because it decisively outperforms independent classifiers at this sample size, where the intervals overlap, but because its averaged probabilities suit the calibrated safety layer to come. Third, the missingness-as-signal hypothesis was tested and largely not supported on this cohort: the confirmatory-test missingness pattern produced no detectable leakage, contrary to the natural worry, and the generic-laboratory missingness pattern offered at most a marginal and inconsistent effect rather than a dependable lab-free signal. The laboratory values themselves moved only the malaria cell and left the decisive dengue cell no better, which makes the lab-free symptoms model the honest operating point, since the one regime that helps depends on tests the surge setting cannot supply. Reporting this as a near-null result rather than a missingness-as-signal success is the more defensible reading of the evidence.

Two questions raised in Section 2.6 deliberately remain open because they require the trained model now in hand. Whether the two health districts are separable to the point that a symptoms-only model trained on one fails on the other, and how a per-label uncertainty estimate can be calibrated so that dangerous cases are referred rather than missed, are taken up in Section 5. The chain-ensemble model and the leak-free fold pipeline defined here are the instruments those analyses use.
""")

md(r"""---

# 5. Evaluation

> The modeling phase established that the multi-label formulation recovers detection the single-label baseline discards, but it did so on patients drawn from one mixed population and it reported point estimates without any statement of when the model should be trusted on an individual patient. The evaluation phase confronts both gaps. It first measures whether a symptoms-only model trained in one health district survives a move to the other, which is the honest test of generalisation for a tool meant to travel. It then equips the model with a calibrated per-label uncertainty estimate so that a dangerous diagnosis is referred for confirmation rather than silently missed. Read together, these two analyses convert the moderate recall numbers of Section 4 into the operating logic of a triage assistant, which is the deliverable this project actually targets, and they are assessed against the business and machine-learning success criteria fixed in Section 1.
""")

md(r"""## 5.1 What Evaluation Must Establish

The success criteria recorded in Section 1 are not a single accuracy threshold, because the deployment goal is not a detector that announces a diagnosis. The goal is a triage and referral assistant for a low-resource surge setting, whose value lies in prioritising which patients to refer for a confirmatory test and in being honest when it cannot decide. Three criteria follow from that framing and structure this section. The first is generalisation: a tool intended to be carried between facilities must be tested across facilities, not only within a pooled sample, because Section 2.5.1 already showed the cohort is two populations with markedly different disease mixes. The second is calibrated uncertainty: the model must attach to each patient a statement of how confident it is, with a guarantee that holds at a stated error rate, so that a clinician can act on the rare confident call and refer the uncertain one. The third is decision safety: the uncertainty must be turned into an explicit referral rule whose costs are stated, so that the failure mode is an over-cautious referral rather than a missed dangerous disease. To these three a fairness screen is added, since a safety behaviour that protected some patients more than others would be unacceptable in a humanitarian tool, and the equity of the referral rule is therefore audited across age, sex, and district. Each of the following subsections addresses one of these concerns in turn, and the section closes by reading the results against the original objectives. The chain-ensemble model and the leak-free fold pipeline from Section 4 are reused without modification, so that the evaluation describes the same model the project would deploy.
""")

# --- 5.2 Site-aware LOSO + ROC overlay ---------------------------------------
md(r"""## 5.2 Generalisation Across Health Districts

The cross-validation in Section 4 balanced the joint label distribution across folds, which answers how the model performs on unseen patients from the same mixed population. It does not answer the question that matters for deployment, which is whether a model fitted where one disease mix prevails still works where a different mix prevails. Section 2.5.1 established that the two health districts differ sharply, with dengue far more common in one and malaria still more dominant in the other, and that a classifier can separate the sites almost perfectly from the clinical record alone. That distribution shift is exactly the condition under which a model can score well in pooled validation and fail on deployment, because the pooled estimate quietly rewards the model for having seen both populations during training. The honest test is leave-one-site-out validation, in which the model is trained on one district and evaluated on the other it has never seen. The grouped cross-validation scheme defined in Section 4.2 implements precisely this, holding out an entire site per fold, and the comparison below places the pooled and leave-one-site-out estimates of dengue detection side by side so the cost of the shift is visible.
""")
code(r'''# Pooled (label-stratified) vs leave-one-site-out OOF probabilities for the same model
# and the same symptoms-only regime. The only thing that changes is the fold assignment.
from sklearn.metrics import roc_curve, roc_auc_score

splits_loso = site_splits()
P_ecc_pool = P_ecc                                   # reuse the stratified OOF from Section 4.4
P_ecc_loso = oof_probabilities(fit_ecc, REGIMES["symptoms"], splits_loso)

di = LABELS.index("dengue")
yd = Yv[:, di]
auc_pool = roc_auc_score(yd, P_ecc_pool[:, di])
auc_loso = roc_auc_score(yd, P_ecc_loso[:, di])

def dengue_recall(P, thr=0.5):
    return recall_score(yd, (P[:, di] >= thr).astype(int), zero_division=0)

print("Dengue detection, same model and features, two validation schemes:")
print(f"  pooled (mixed population)      : ROC-AUC {auc_pool:.2f}   recall@0.5 {dengue_recall(P_ecc_pool):.1%}")
print(f"  leave-one-site-out (transfer)  : ROC-AUC {auc_loso:.2f}   recall@0.5 {dengue_recall(P_ecc_loso):.1%}")
print(f"  drop in ROC-AUC on transfer    : {auc_pool - auc_loso:.2f} (pooled minus transfer)")

# The leave-one-site-out scheme has only two folds (one per site), so the transfer AUC is
# reported per fold as well: a single pooled number would hide which direction of transfer
# fails. An AUC below 0.5 means the model ranks dengue patients systematically BELOW
# non-dengue under the shift, which is a symptom of the distribution shift rather than a
# rank-inversion bug, since the same model scores 0.78 on the matched population.
print("  per-fold transfer AUC (train on one district, test on the other):")
for tr, te in splits_loso:
    held = pd.unique(site.values[te])
    if Yv[te, di].sum() > 0:
        a = roc_auc_score(Yv[te, di], P_ecc_loso[te, di])
        print(f"    test on {str(held[0]):6s} (n={len(te)}, dengue={int(Yv[te, di].sum())}): AUC {a:.2f}")

# Per-site sanity: how much does each district's dengue prevalence differ?
for s in sorted(site.unique()):
    m = (site.values == s)
    print(f"    site {s}: n={m.sum():3d}, dengue prevalence {yd[m].mean():.1%}")
''')
md(r"""The contrast is the central honesty result of the evaluation. Detection of dengue that looks workable under pooled validation degrades sharply when the model must transfer to a district it has not seen, because the symptom signature the model learned is entangled with the disease mix of the training district rather than being a portable marker of dengue. Two complementary numbers make the degradation precise and should be read together. The pooled out-of-fold area under the curve is 0.78, while the area computed by pooling the predictions of the two transfer models across all patients falls to 0.37, below chance. The within-fold areas, computed inside each held-out district separately, are about 0.58 to 0.60, well below the pooled 0.78 but above chance. The gap between the two transfer figures is itself the finding: each transfer model still ranks patients a little better than chance inside the single district it is tested on, but the scores produced by a model trained on one district are not comparable to those produced by a model trained on the other, so once the two are pooled onto a common axis the ranking breaks down entirely. That non-comparability of scores across districts is exactly what defeats a fixed operating threshold in deployment, and it is a sharper statement of the same problem than a single summary number would give. This is not a defect of the particular model; it is the unavoidable consequence of training on a population that differs from the deployment population, and it would afflict any learner given these features. Reporting only the pooled 0.78 would therefore overstate what the tool can be expected to do in a new facility, which is exactly the kind of silent over-claim this project set out to avoid. The figure overlays the pooled and pooled-transfer receiver operating characteristic curves so the collapse can be read directly, and it carries a single, decisive message for deployment: a symptoms-only model must not be presented as a cross-region dengue detector. The constructive response is not to abandon the model but to change what is asked of it, from naming a diagnosis to flagging a referral under an explicit statement of uncertainty, which is the subject of the next two subsections.
""")
code(r'''# Figure 5.2: ROC overlay, pooled vs leave-one-site-out, for the dengue cell.
fpr_p, tpr_p, _ = roc_curve(yd, P_ecc_pool[:, di])
fpr_l, tpr_l, _ = roc_curve(yd, P_ecc_loso[:, di])

figr = go.Figure()
figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                          line=dict(color="#BBB", dash="dot", width=1.5),
                          name="chance", hoverinfo="skip"))
figr.add_trace(go.Scatter(x=fpr_p, y=tpr_p, mode="lines",
                          line=dict(color=PRIMARY, width=3),
                          name=f"pooled, same population (AUC {auc_pool:.2f})",
                          hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>pooled</extra>"))
figr.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode="lines",
                          line=dict(color=NEGATIVE, width=3),
                          name=f"leave-one-site-out, transfer (AUC {auc_loso:.2f})",
                          hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>transfer</extra>"))
figr.update_layout(xaxis_title="false positive rate", yaxis_title="true positive rate",
                   xaxis=dict(range=[-0.02, 1.02]), yaxis=dict(range=[-0.02, 1.02]),
                   legend=dict(x=0.97, y=0.04, xanchor="right", yanchor="bottom"))
show_interactive(figr, "fig_loso_roc", height=520, width=720,
                 title="Dengue detection collapses when the model must cross health districts")
''')

md(r"""The collapse on transfer raises an obvious question: if the symptom model does not carry dengue across the two districts, what is it carrying instead? The answer is the district itself. If the recorded features encode which facility a patient attended, then a model can score well in pooled validation by quietly inferring the site and exploiting its disease mix, which is precisely the shortcut that fails on transfer. This is testable directly. A classifier trained to predict the health district from the full clinical record, rather than the diagnosis, measures how recoverable the site is from the features and therefore how large the domain gap is, a quantity closely related to the divergence between the two populations. The figure below reports that site classifier and shows its per-patient scores, so that the near-perfect separation can be seen rather than asserted.
""")
code(r'''# Domain classifier: how recoverable is the health district from the full feature record?
# A near-perfect site classifier means the features encode geography, which is exactly the
# shortcut that lets pooled CV look good and then fails on cross-district transfer.
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

site_y = (site.values == site.value_counts().index[0]).astype(int)   # one site vs the other
site_name_1 = site.value_counts().index[0]
dom_scores = cross_val_predict(
    make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                  LogisticRegression(max_iter=2000, C=0.5)),
    X.values, site_y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    method="predict_proba")[:, 1]
auc_site = roc_auc_score(site_y, dom_scores)
print(f"Site recoverable from the full feature record at cross-validated AUC = {auc_site:.2f}")
print(f"(an AUC near 1.0 means the two districts are almost perfectly separable from the "
      f"clinical features alone, i.e. a large domain gap)")

# Figure: per-patient site-classifier score, one strip per district, to show the separation.
fig, ax = plt.subplots(figsize=(8.4, 3.2))
ax.set_axisbelow(True)
_rng_s = np.random.default_rng(SEED)   # jitter is cosmetic but seeded from the global SEED
for g, nm, col in [(1, str(site_name_1), SITE_A), (0, "other district", SITE_B)]:
    xv = dom_scores[site_y == g]
    yv = g + (_rng_s.random(len(xv)) - 0.5) * 0.55
    ax.scatter(xv, yv, s=18, color=col, alpha=0.6, edgecolor="white", linewidth=0.4,
               label=nm, zorder=3)
ax.axvline(0.5, color="#999", ls=":", lw=1, zorder=2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["other district", str(site_name_1)])
ax.set_xlim(0, 1); ax.set_ylim(-0.6, 1.6)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)) if False else None
style_ax(ax, xlabel="site-classifier score", grid_axis="x")
ax.annotate(f"AUC = {auc_site:.2f}", (0.5, 0.5), ha="center", va="center", fontsize=10,
            color="#222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDD"))
ax.set_title("The features encode geography: the two districts are almost perfectly separable",
             loc="center", pad=8, fontsize=13)
plt.tight_layout(); plt.show()
''')
md(r"""The site classifier separates the two districts almost perfectly, which confirms the mechanism behind the transfer collapse: the clinical record carries a strong signature of where the patient was seen, and a naive model rewarded on pooled data will lean on that signature instead of on a portable disease marker. A complementary and more intuitive way to see the same fact is to ask whether patients fall into clean symptom-defined groups at all. Projecting the binary symptom matrix to two dimensions, by a singular value decomposition of the centred indicator matrix in the spirit of multiple correspondence analysis, places each patient in a plane where proximity means similar symptoms. Colouring that plane first by disease profile and then by district answers the question visually: if symptoms defined the diseases, the disease colours would separate, and if symptoms instead tracked geography, the site colours would separate. The figure shows the latter. This is an exploratory projection and no formal cluster claim rests on it, but it makes the central obstacle of the whole study legible, namely that the dominant axis of symptom variation in this cohort is the health district rather than the diagnosis, which is exactly why a symptoms-only model generalises poorly across sites and why the safe response is referral under uncertainty rather than a confident label.
""")
code(r'''# Phenotype map: project the binary symptom matrix to 2D and colour by disease vs site.
SYM_COLS = [c for c in feature_groups["symptom"] if c in X.columns]
Xs = X[SYM_COLS].fillna(0).values.astype(float)
Xc = Xs - Xs.mean(0, keepdims=True)
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
# The SVD is only defined up to a sign per component, which can mirror the plot between
# platforms; fix the sign deterministically so the figure is reproducible.
for k in range(2):
    if U[:, k].sum() < 0:
        U[:, k] *= -1; Vt[k, :] *= -1
coords = U[:, :2] * sv[:2]
ev = (sv**2 / (sv**2).sum())[:2]

def profile_name(i):
    d, m = Yv[i, LABELS.index("dengue")], Yv[i, LABELS.index("malaria")]
    if d == 1 and m == 0: return "non-malarial dengue"
    if d == 1:            return "malaria + dengue"
    if m == 1 and Yv[i].sum() == 1: return "malaria only"
    return "other"
prof = np.array([profile_name(i) for i in range(len(Yv))])
pcol = {"malaria only": SECONDARY, "malaria + dengue": PRIMARY,
        "non-malarial dengue": NEGATIVE, "other": WARNING}
site_arr = site.values

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.16})
for p, c in pcol.items():
    m = prof == p
    a1.scatter(coords[m, 0], coords[m, 1], s=22, color=c, label=p, alpha=0.8,
               edgecolor="white", linewidth=0.4, zorder=3 if "dengue" in p else 1)
a1.set_title("coloured by disease profile", loc="center", fontsize=12)
a1.legend(fontsize=7.5, loc="best")
for s_, c_ in [(site.value_counts().index[0], SITE_A), (site.value_counts().index[1], SITE_B)]:
    m = site_arr == s_
    a2.scatter(coords[m, 0], coords[m, 1], s=22, color=c_, label=str(s_), alpha=0.8,
               edgecolor="white", linewidth=0.4)
a2.set_title("coloured by health district", loc="center", fontsize=12)
a2.legend(fontsize=8)
for a in (a1, a2):
    style_ax(a, xlabel=f"dimension 1 ({ev[0]:.0%} variance)",
             ylabel=f"dimension 2 ({ev[1]:.0%} variance)", grid_axis="none")
    a.grid(True, alpha=0.3)
fig.suptitle("Symptom variation tracks the health district, not the diagnosis",
             y=1.0, fontsize=15, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.show()
print(f"First two symptom components explain {ev.sum():.0%} of the variance "
      f"(low, so no single dominant symptom axis); the visible split in the right "
      f"panel is the health district.")
''')

md(r"""## 5.2.1 Convergence with Two Independent African Cohorts

The leave-one-site-out result of Section 5.2 is a statement about generalisation within one study. Its credibility is strengthened if the same qualitative findings appear in cohorts collected by other teams in other countries, and weakened if they do not. Two independent paediatric febrile-illness cohorts allow that check. The first is the western and coastal Kenyan cohort already introduced in Section 4.4.1, with dengue confirmed by polymerase chain reaction and malaria by rapid test. The second is a Tanzanian paediatric fever cohort in which malaria was confirmed by both blood smear and rapid test and typhoid by rapid test (multi-disease febrile-illness study; CC BY). Neither cohort is merged with the cohort under study or used to train any model reported here; each is analysed on its own to ask whether the central pattern of this project, that symptoms separate the common disease moderately while leaving the minority and dangerous diseases poorly separable, recurs outside Burkina Faso. The two cohorts are complementary in what they can corroborate: the Kenyan data speaks to dengue, the hardest and most decision-relevant cell of the primary cohort, while the Tanzanian data speaks to malaria and typhoid. Together they cover three of the four diseases; yellow fever has no comparable public cohort and remains, here as throughout, a descriptive case only.
""")
code(r'''# External convergence: train a simple symptom-only classifier within each external cohort
# (5-fold out-of-fold AUC) and compare the resulting separability to the primary cohort's
# own symptom-only signal. Both cohorts are used only for this corroboration; neither is
# merged into the study data or into any model this notebook reports.
from pathlib import Path
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.model_selection import cross_val_predict as _cvp, StratifiedKFold as _SKF
from sklearn.metrics import roc_auc_score as _auc

_cv = _SKF(5, shuffle=True, random_state=SEED)
def _oof_auc(X, y):
    if y.sum() < 8 or (len(y) - y.sum()) < 8:
        return np.nan, int(y.sum()), len(y)
    p = _cvp(_LR(max_iter=1000, class_weight="balanced"), X, y, cv=_cv,
             method="predict_proba")[:, 1]
    return _auc(y, p), int(y.sum()), len(y)

conv = []  # (cohort, disease, auc, n_pos, n)

# Kenya: symptoms (6 groups) -> dengue PCR, and -> malaria RDT.
_KCSV = (Path("external_data") / "Western and coastal Kenya sick visit data 2019-2022"
         / "Western_and_coastal_Kenya_sick_visit_data_2019_2022.csv")
if _KCSV.exists():
    kf = pd.read_csv(_KCSV)
    _S6 = ["head_symptoms_sick", "eent_symptoms_sick", "chest_breath_symptoms_sick",
           "stomach_symptoms_sick", "muscles_symptoms_sick", "skin_blood_symptoms_sick"]
    Xk = kf[_S6].apply(lambda s: pd.to_numeric(
        s.replace({"Unsure/Don'tknow": np.nan}), errors="coerce"))
    ok = Xk.notna().all(axis=1)
    for col, dis in [("ufi_zcd_dengue_result", "dengue"), ("rdt_malaria_results_sick", "malaria")]:
        if col == "ufi_zcd_dengue_result":
            lab = kf[col].isin(["Positive", "Negative"]); y = (kf[col] == "Positive").astype(int)
        else:
            lab = kf[col].isin([0.0, 1.0]); y = (kf[col] == 1.0).astype(int)
        m = (ok & lab).values
        a, npos, ntot = _oof_auc(Xk[m].values, y[m].values)
        conv.append(("Kenya", dis, a, npos, ntot))

# Tanzania: individual symptoms -> malaria (blood smear) and -> typhoid (RDT).
_TZ = Path("external_data") / "tanzania_fever_full.parquet"
if _TZ.exists():
    tz = pd.read_parquet(_TZ)
    _TS = ["fever", "headache", "cough", "vomiting", "diarrhea", "jaundice", "rashother"]
    Xt = tz[_TS].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    okt = Xt.notna().all(axis=1)
    for col, dis in [("bsmalaria", "malaria"), ("rdttyphoid", "typhoid")]:
        lab = tz[col].isin(["P", "N"]); y = (tz[col] == "P").astype(int)
        m = (okt & lab).values
        a, npos, ntot = _oof_auc(Xt[m].values, y[m].values)
        conv.append(("Tanzania", dis, a, npos, ntot))

if conv:
    cv_tbl = pd.DataFrame(conv, columns=["cohort", "disease", "symptom AUC", "n positive", "n"])
    ipd.display(cv_tbl.round(3))
    print("Reading: malaria is moderately separable from symptoms in every cohort, while "
          "dengue and typhoid are weakly separable, matching the primary cohort. The pattern "
          "is not an artefact of one dataset.")
else:
    print("External cohorts not found; the convergence check is skipped (the notebook's own "
          "conclusions do not depend on it).")
''')
md(r"""The external evidence converges with the primary cohort rather than contradicting it. In both external cohorts malaria, the common disease, is moderately separable from symptoms alone, with an out-of-fold area under the curve close to 0.71 in the Tanzanian data and to 0.62 to 0.73 in the Kenyan data depending on the target, which is the same regime of moderate but usable malaria signal seen within Burkina Faso. The diseases that matter most for safe triage behave as the primary cohort warned they would: dengue in the Kenyan data is only weakly separable from its six grouped symptoms, and typhoid in the Tanzanian data sits barely above chance, mirroring the near-nesting of typhoid within malaria and the weak single-symptom ceiling established in Sections 2.3 and 2.5. The practical reading is twofold. First, the central difficulty this project confronts, that the dangerous minority diseases cannot be cleanly named from symptoms, is a property of febrile-illness data across at least three African settings and not a deficiency peculiar to this cohort, which is exactly why an uncertainty-aware referral tool rather than a confident detector is the appropriate response. Second, the comparison is necessarily coarse, because the Kenyan symptoms are pre-aggregated into groups and the Tanzanian feature set differs from the primary one, so these numbers corroborate the qualitative pattern rather than constitute a transferred model or a like-for-like benchmark, a limitation that Section 6 turns into the central design question of the generalisation roadmap.
""")

# --- 5.3 Per-label conformal -------------------------------------------------
md(r"""## 5.3 Calibrated Per-Label Uncertainty

A point prediction, even an accurate one, does not tell a clinician when it can be trusted, and Section 5.2 has just shown that trust is exactly what cannot be assumed when the population shifts. The remedy is to replace the single probability per disease with a prediction set whose error rate is controlled at a level chosen in advance. The instrument is split conformal prediction applied separately to each disease, sometimes called the Mondrian or class-conditional variant, which calibrates a threshold on the model's scores using a held-out portion of the data and then guarantees, under the assumption that the calibration and test patients are exchangeable, that the true label falls inside the set at least as often as the chosen confidence demands. Applied per disease, the procedure produces for each patient a set of diseases that cannot be ruled out at the requested confidence, which is the natural language of triage: a small set is a confident call, and a large set is an explicit admission that a confirmatory test is needed. The guarantee is honoured only for the two diseases the cohort can support, malaria and dengue, since yellow fever and typhoid have too few positive cases to calibrate a stable threshold, and those two are reported descriptively. The calibration uses an out-of-fold split so that no patient is scored by a model that has seen them, and the coverage is verified empirically against the nominal target.
""")
code(r'''# Split-conformal, per label (Mondrian / class-conditional). For each label we form a
# nonconformity score on a calibration split and threshold it so that the prediction set
# covers the true label at the requested rate. We reuse the pooled OOF probabilities as the
# scores, then split them once into calibration and test halves, stratified within each label.
from numpy.random import default_rng as _rng

ALPHA = 0.10                                   # target miscoverage: 90% sets should contain truth
CONF_LABELS = ["malaria", "dengue"]            # guarantee claimed only for these (Section 1 honesty rule)

rng_c = _rng(SEED)
n = len(Yv)
perm = rng_c.permutation(n)
cal_idx, test_idx = perm[: n // 2], perm[n // 2 :]   # one 50/50 calibration/test split

def conformal_label(P, j, cal_idx, test_idx, alpha=ALPHA):
    """Class-conditional split conformal for one label.
    Returns, for the test rows: lower-inclusion (label is IN the set) booleans for y=1 and y=0
    decided by separate thresholds calibrated on positives and negatives, and the empirical
    coverage on the test split."""
    s = 1.0 - P[:, j]                          # nonconformity for the positive class: 1 - p(disease)
    # Calibrate a threshold on the calibration positives so that 1-alpha of them are retained.
    cal_pos = cal_idx[Yv[cal_idx, j] == 1]
    if len(cal_pos) < 10:
        return None
    qhat = np.quantile(s[cal_pos], 1 - alpha, method="higher")   # conformal quantile
    in_set = (s[test_idx] <= qhat)             # disease kept in the set when score is conforming
    test_pos = Yv[test_idx, j] == 1
    cov = in_set[test_pos].mean()              # empirical coverage on true positives
    return dict(qhat=qhat, in_set=in_set, coverage=cov,
                set_rate=in_set.mean(), n_cal_pos=len(cal_pos),
                n_test_pos=int(test_pos.sum()))

def wilson_ci(k, nobs, z=1.96):
    """Wilson score interval for a proportion k/nobs; honest uncertainty at small n."""
    if nobs == 0:
        return (np.nan, np.nan)
    p = k / nobs
    d = 1 + z**2 / nobs
    centre = (p + z**2 / (2 * nobs)) / d
    half = (z * np.sqrt(p * (1 - p) / nobs + z**2 / (4 * nobs**2))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))

print(f"Per-label split conformal at {int((1-ALPHA)*100)}% target coverage "
      f"(calibration n={len(cal_idx)}, test n={len(test_idx)}):")
conf = {}
for l in CONF_LABELS:
    r = conformal_label(P_ecc, LABELS.index(l), cal_idx, test_idx)
    conf[l] = r
    assert r is not None, f"{l} should be well powered for conformal calibration"
    npos = r["n_test_pos"]
    lo, hi = wilson_ci(int(round(r["coverage"] * npos)), npos)
    print(f"  {DISP[l]:8s}: empirical coverage on true cases {r['coverage']:.0%} "
          f"[95% Wilson {lo:.0%}, {hi:.0%}] on n={npos} test positives "
          f"(target {int((1-ALPHA)*100)}%), disease kept in set for {r['set_rate']:.0%} of test patients, "
          f"calibration positives {r['n_cal_pos']}")
print("\nDescriptive only (too few positives to calibrate a stable threshold):")
for l in ["typhoid", "yellow_fever"]:
    print(f"  {DISP[l]:12s}: {int(Yv[:, LABELS.index(l)].sum())} positives in the full cohort")
''')
md(r"""The empirical coverage sits close to the nominal target for both calibrated diseases, malaria slightly above at ninety two percent and dengue slightly below at eighty nine percent. Neither departure should be over-read: dengue coverage is estimated from only about twenty seven true positives in the test split, so its ninety five percent Wilson interval is wide and comfortably contains the ninety percent target, and the one point shortfall is sampling noise rather than a systematic failure of the guarantee. The honest reading is that both diseases show coverage near the nominal level rather than gross over or under coverage, which is what the procedure is meant to deliver. The practical difference between the two diseases is informative in its own right. For malaria the prediction set is usually small, because the model is confident and the disease is common, so the set typically resolves to a single confident call. For dengue the set is retained far more often relative to how many patients actually have dengue, which is the conformal procedure correctly reporting that the symptom signal is weak: rather than committing to a low-probability dengue call or silently dropping it, the method keeps dengue in the set and thereby marks the patient as one who should be referred for a confirmatory test. This is the mechanism by which the moderate dengue recall of Section 4 becomes clinically safe. The model is not asked to be right about dengue, which the evidence says it cannot reliably be from symptoms alone; it is asked to know when it does not know, and the conformal layer gives that knowledge a guarantee. One limitation must be stated plainly: the exchangeability assumption underlying the guarantee is violated under the cross-district shift quantified in Section 5.2, so the stated coverage holds within the mixed population and should be read as an upper bound on what to expect in a genuinely new facility, where the calibration would have to be refreshed on local data. The next subsection turns these sets into an explicit referral rule.
""")

# --- 5.4 Cost-sensitive abstention / referral --------------------------------
md(r"""## 5.4 From Prediction Sets to a Referral Rule

A prediction set is only useful if it drives an action, and in a triage setting the action is a three-way decision rather than a diagnosis. When the set contains a single disease the model is making a confident call and the patient can be managed on that basis. When the set is empty the model is confident that none of the calibrated diseases is present, which in this malaria-saturated cohort is itself a signal worth noting. When the set contains more than one disease, or contains dengue specifically, the model is declaring that it cannot rule out a dangerous diagnosis, and the safe response is to refer the patient for the confirmatory test that would resolve the ambiguity. The rule is anchored on dengue for a specific and honestly stated reason. Dengue is not the only dangerous disease in this cohort, since yellow fever in particular can be fatal, but dengue is the one dangerous and weakly separable disease for which this cohort provides enough positive cases to support a calibrated coverage guarantee, as Section 5.3 established. Yellow fever and typhoid are too rare here to calibrate a stable prediction set, so they are reported descriptively and do not by themselves drive the automated trigger; in a setting where those diseases are common the rule would need its own calibration for them, and clinical severity signs remain the clinician's guide in the meantime. This is a deliberately asymmetric rule, because the costs are asymmetric: referring a patient who turns out not to have dengue consumes a test and a clinician's time, whereas failing to refer a patient who does have dengue risks a missed case of a disease that can progress to a severe haemorrhagic form. The rule below makes that asymmetry explicit by counting the two kinds of error separately rather than collapsing them into a single accuracy figure, and it reports what fraction of patients the assistant would refer, since a triage tool that refers everyone is as useless as one that refers no one.
""")
code(r'''# Turn the per-label conformal sets into a referral decision on the test split.
# Referral is triggered when dengue cannot be ruled out (dengue stays in its conformal set).
# Dengue anchors the rule not because it is uniquely dangerous but because it is the only
# dangerous, weakly-separable disease this cohort can calibrate a guarantee for (Section 5.3);
# yellow fever and typhoid are too rare here to calibrate and do not drive the trigger.
di_, mi_ = LABELS.index("dengue"), LABELS.index("malaria")
in_dengue = conf["dengue"]["in_set"]            # dengue retained in the set (per test row)
in_malaria = conf["malaria"]["in_set"]

y_test = Yv[test_idx]
true_dengue  = y_test[:, di_] == 1
true_malaria = y_test[:, mi_] == 1

refer = in_dengue                               # the referral trigger
referral_rate = refer.mean()

# Safety accounting against the dengue ground truth on the test split.
caught   = (refer & true_dengue).sum()          # dengue referred (good)
missed   = (~refer & true_dengue).sum()         # dengue NOT referred (the costly error)
overref  = (refer & ~true_dengue).sum()         # referred without dengue (the tolerable error)
dengue_n = int(true_dengue.sum())

print(f"Referral rule: refer when dengue cannot be ruled out (test split n={len(test_idx)})")
print(f"  patients referred           : {refer.sum()}  ({referral_rate:.0%} of the test cohort)")
print(f"  true dengue cases referred  : {caught} of {dengue_n}  (sensitivity of the referral path {caught/max(dengue_n,1):.0%})")
print(f"  true dengue cases MISSED     : {missed}   <- the error the asymmetric rule minimises")
print(f"  referred without dengue      : {overref}  (tolerated cost: a confirmatory test)")

# Contrast with the naive single-threshold detector at 0.5 on the same test rows:
naive_refer = (P_ecc[test_idx, di_] >= 0.5)
naive_caught = (naive_refer & true_dengue).sum()
print(f"\nFor comparison, a naive 0.5 dengue detector on the same rows refers "
      f"{naive_refer.sum()} patients and catches {naive_caught} of {dengue_n} dengue cases.")
print("The conformal referral path is calibrated to a coverage guarantee; the naive threshold is not.")
''')
md(r"""The referral rule realises the triage positioning that has guided the whole project. By referring every patient for whom dengue cannot be confidently ruled out, the assistant converts the model's acknowledged weakness on dengue into a controlled and inspectable behaviour, trading a quantified number of unnecessary confirmatory tests for a reduction in the silent misses that are the genuinely dangerous outcome. The accounting separates the tolerable error from the costly one rather than averaging them, which is the appropriate stance when one error consumes a test and the other risks a life. It is important to state what this rule is and is not. It is a transparent decision layer on top of a calibrated model, suitable for prioritising scarce confirmatory testing during a surge, and it degrades gracefully because its worst ordinary failure is an over-referral. It is not a diagnosis, it does not remove the need for the confirmatory test it recommends, and its coverage guarantee is inherited from Section 5.3 and therefore subject to the same exchangeability caveat under cross-district transfer. Concretely, because Section 5.2 showed dengue ranking collapses on a genuinely new district, the referral path's sensitivity reported here should be treated as a within-population figure that would require local recalibration before deployment elsewhere, and the same recalibration that refreshes the conformal threshold would refresh this rule. Within those honest limits it is the component that makes a moderately accurate symptom model defensible as a humanitarian triage aid, because it ensures the model fails toward caution rather than toward silence.
""")

# --- 5.6 Fairness audit ------------------------------------------------------
md(r"""## 5.6 Fairness of the Referral Behaviour

A triage assistant is only acceptable if its safety behaviour does not quietly favour one group of patients over another, and the cohort description in Section 2 flagged three axes along which that must be checked. The first is age, because the population is heavily weighted toward children and a tool that referred adults more readily than children, or the reverse, would distribute the protective benefit of referral unevenly across the most vulnerable. The second is sex, the standard equity axis. The third is the health district itself, which Sections 5.2 and the phenotype map have already shown to be the dominant source of variation, so any disparity in referral behaviour is most likely to appear there. The audit is descriptive rather than a formal test, because several subgroups are small and the cohort is underpowered for a stringent fairness claim, but the relevant quantities can still be reported honestly. For each subgroup we report how often the referral rule fires and, more importantly, the sensitivity of the referral path to true dengue within that subgroup, since an equitable assistant should catch dangerous cases at a similar rate everywhere even if its overall referral rate differs with the local disease mix.
""")
code(r'''# Fairness audit of the referral rule across age band, sex, and health district.
# Computed on the conformal TEST split so it reflects the deployed referral behaviour.
# refer / true_dengue / test_idx were defined in Section 5.4.
age_band_test = X["age_band"].values[test_idx]          # 0=under 5, 1=5-15, 2=15+
female_test   = X["is_female"].values[test_idx]
site_test     = site.values[test_idx]

AGE_NAMES = {0.0: "under 5", 1.0: "5 to 15", 2.0: "15 and over"}

def subgroup_row(name, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    dn = int((true_dengue & mask).sum())                 # true dengue in this subgroup (test split)
    refer_rate = refer[mask].mean()
    caught = int((refer & true_dengue & mask).sum())     # true dengue referred in the subgroup
    # sensitivity of the referral path to dengue within the subgroup
    sens = caught / dn if dn else np.nan
    return dict(subgroup=name, n=n, dengue_n=dn, caught=caught,
                referral_rate=refer_rate, dengue_sensitivity=sens)

MIN_POWERED = 5   # below this many true dengue cases a per-group sensitivity is not interpretable

rows = []
for code_, nm in AGE_NAMES.items():
    rows.append(subgroup_row(f"age: {nm}", age_band_test == code_))
for val, nm in [(1.0, "sex: female"), (0.0, "sex: male")]:
    rows.append(subgroup_row(nm, female_test == val))
for s in sorted(pd.unique(site_test)):
    rows.append(subgroup_row(f"site: {s}", site_test == s))

fair = pd.DataFrame([r for r in rows if r is not None]).set_index("subgroup")

# Display the sensitivity as caught/dengue_n so the thinness of small cells is visible, and
# suppress the percentage to "underpowered" when fewer than MIN_POWERED true dengue cases exist.
def _sens_cell(row):
    if row["dengue_n"] == 0:
        return "n/a"
    frac = f"{row['caught']}/{row['dengue_n']}"
    if row["dengue_n"] < MIN_POWERED:
        return f"{frac} (underpowered)"
    return f"{row['dengue_sensitivity']:.0%} ({frac})"

disp_fair = fair.assign(
    referral_rate=lambda d: (d["referral_rate"]*100).round(0).astype(int).astype(str)+"%",
    dengue_sensitivity=lambda d: d.apply(_sens_cell, axis=1))[
    ["n", "dengue_n", "referral_rate", "dengue_sensitivity"]]
ipd.display(disp_fair)
print(f"Read with care: this is a descriptive screen across {len(fair)} subgroups, not a set of "
      f"powered hypothesis tests, so no multiple-comparison correction is applied or implied. "
      f"Per-group dengue sensitivity is shown as caught/total and is marked underpowered when "
      f"fewer than {MIN_POWERED} true dengue cases fall in the group; such cells should not be "
      f"compared against the others.")
''')
md(r"""The audit returns a reassuring but appropriately hedged picture. The referral rate varies most across the two health districts, which is expected and not in itself a fairness problem, because the districts genuinely differ in dengue prevalence and a rule that refers more where dengue is more common is behaving correctly rather than inequitably. The safety-critical quantity is the sensitivity of the referral path to true dengue within each subgroup. By sex the two estimates differ by roughly ten percentage points, which on twelve and fifteen true dengue cases respectively is well within sampling noise and does not constitute demonstrable bias, so the rule is best described as not detectably unequal by sex rather than exactly balanced. By age the older bands are well protected, while the lowest sensitivity falls in the under-five band, which on inspection rests on only two true dengue cases in that group on the test split, of which the rule referred one, so the apparent gap is dominated by sampling noise rather than a demonstrated bias and cannot be read as a reliable disparity at this sample size. The honest statement is therefore narrow but defensible: the audit finds no balanced-and-powered evidence of inequity by sex, and the one age band that looks worse is exactly the one the cohort is too small to judge, which is a call for more data in that group rather than a finding against the rule. Two honest qualifications accompany the reading. Several subgroups contain only a handful of true dengue cases, so the per-group sensitivities are estimated with wide uncertainty and a small disparity could be hidden by sampling noise; the audit is therefore a screen for gross inequity, which it does not find, rather than a guarantee of fairness, which this cohort cannot support. And the clearest disparity, the district one, is the same distribution shift documented throughout Section 5, which reinforces that local recalibration, not a global threshold, is the responsible path to deployment.
""")

# --- 5.7 Missingness-signal map (model-free, consistent with the 4.5 null) ----
md(r"""## 5.7 A Model-Free View of the Missingness Signal

Section 4.5 reached a clear and slightly counterintuitive conclusion, that the pattern of which laboratory tests were ordered carries essentially no usable signal for the decisive dengue cell once it is placed inside a properly cross-validated multi-label model. It is worth confirming that this null is not an artefact of the model by inspecting the raw association directly, model-free, and reporting it transparently alongside the disease co-occurrence structure that frames the whole problem. The figure below pairs the two. On the left is the co-occurrence matrix of the four diseases, whose diagonal is each disease's prevalence and whose off-diagonal cells are co-infection counts, making visible once more that almost everything funnels through malaria. On the right is the missingness-as-signal view: each laboratory field that has at least eight patients in both the missing and the present state, so that a rate can be compared at all, is plotted by how often it is missing against how much its being missing shifts the raw dengue rate, with the generic and confirmatory fields distinguished by colour, and the very sparse fields that cannot support a stable ratio are omitted and counted in the output. A field sitting far from the no-shift line would be one where the mere fact that a test was not ordered moves the dengue rate, which is the descriptive footprint of a missingness signal. The point of showing this after the modelling result, rather than before, is to let the reader see that even the raw associations are modest and scattered around no shift, which is consistent with the modelled null rather than in tension with it.
""")
code(r'''# Model-free missingness-as-signal map + disease co-occurrence, shown to corroborate the
# Section 4.5 null: raw dengue-rate shifts by missingness are modest and scattered around 1.0.
from plotly.subplots import make_subplots

L = LABELS
Yb = Y[L].astype(int)
co = np.zeros((len(L), len(L)), int)
for i, a in enumerate(L):
    for j, b in enumerate(L):
        co[i, j] = int(((Yb[a] == 1) & (Yb[b] == 1)).sum())
disp = [DISP[l] for l in L]
celltext = [[("diagonal: prevalence" if i == j else "co-infection")
             for j in range(len(L))] for i in range(len(L))]
heat = go.Heatmap(z=co, x=disp, y=disp, colorscale="Reds", showscale=False,
                  text=co, texttemplate="%{text}", textfont=dict(size=13), customdata=celltext,
                  hovertemplate="%{y} and %{x}: <b>%{z}</b> patients<br>%{customdata}<extra></extra>")

ydv = Y["dengue"].astype(int)
mask_conf_set, mask_gen_set = set(mask_confirmatory), set(mask_generic)
pts = []
n_skipped = 0
for f in M.columns:
    miss = M[f].astype(bool)
    n_miss, n_obs = int(miss.sum()), int((~miss).sum())
    if min(n_miss, n_obs) < 8:        # both arms must be populated to compare a rate
        n_skipped += 1
        continue
    r_miss, r_obs = ydv[miss].mean(), ydv[~miss].mean()
    ratio = (r_miss + 1e-3) / (r_obs + 1e-3)
    base = f[:-6] if f.endswith("__isna") else f
    grp = "confirmatory" if f in mask_conf_set else ("generic" if f in mask_gen_set else "other")
    pts.append((base, miss.mean(), ratio, n_miss, r_miss, r_obs, grp))
PT = pd.DataFrame(pts, columns=["feat", "miss_rate", "ratio", "n_miss", "r_miss", "r_obs", "grp"])
GCOL = {"generic": POSITIVE, "confirmatory": NEGATIVE, "other": SECONDARY}

fig2 = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.12,
                     subplot_titles=("Disease co-occurrence", "Missingness-as-signal"))
fig2.add_trace(heat, row=1, col=1)
xr = [PT["miss_rate"].min(), PT["miss_rate"].max()]
fig2.add_trace(go.Scatter(x=xr, y=[1.0, 1.0], mode="lines", name="no shift (ratio = 1)",
                          line=dict(color="#555", dash="dash", width=1.4), hoverinfo="skip"),
               row=1, col=2)
for g in ["generic", "confirmatory", "other"]:
    d = PT[PT["grp"] == g]
    if d.empty:
        continue
    fig2.add_trace(go.Scatter(
        x=d["miss_rate"], y=d["ratio"], mode="markers", name=g,
        marker=dict(size=9 + 22 * (d["n_miss"] / PT["n_miss"].max()), color=GCOL[g],
                    line=dict(width=1, color="white"), opacity=0.85),
        customdata=np.c_[d["feat"], d["n_miss"], d["r_miss"], d["r_obs"]],
        hovertemplate=("<b>%{customdata[0]}</b> (%{x:.0%} missing, n=%{customdata[1]})<br>"
                       "dengue rate when missing %{customdata[2]:.0%} vs present "
                       "%{customdata[3]:.0%}<br>ratio %{y:.2f}<extra></extra>")), row=1, col=2)
fig2.update_xaxes(title_text="fraction of patients missing this field", tickformat=".0%", row=1, col=2)
fig2.update_yaxes(title_text="dengue-rate ratio (missing over present)", row=1, col=2)
for a in fig2.layout.annotations:
    a.text = f"<b>{a.text}</b>"
fig2.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.53,
                               bgcolor="rgba(255,255,255,0.6)", borderwidth=0))
show_interactive(fig2, "fig_cooccur_missingness", width=980, height=460,
                 title="The raw missingness signal is modest, consistent with the modelled null")
print(f"Of the mask fields, {len(PT)} had both arms populated (at least 8 patients missing and "
      f"8 present) and are plotted (generic={int((PT.grp=='generic').sum())}, "
      f"confirmatory={int((PT.grp=='confirmatory').sum())}); {n_skipped} too-sparse fields were "
      f"omitted because their dengue-rate ratio would be unstable. Most plotted points sit near "
      f"the no-shift line.")
''')
md(r"""The raw view corroborates the modelled conclusion rather than complicating it. Most laboratory fields cluster near the no-shift line, meaning that whether the test was ordered barely moves the raw dengue rate, and the few fields that sit further away are precisely those with the smallest counts, where a couple of patients swing the ratio and the apparent signal is fragile. Crucially, the confirmatory-test fields, the ones that would carry leakage of clinician intent if any field did, do not stand out as a group, which matches the modelled finding that the confirmatory mask produced no detectable gain. Seeing the descriptive and the modelled results agree is more convincing than either alone: the missingness-as-signal hypothesis is not merely unsupported by one particular classifier, it is weak in the raw data as well, so reporting it as a near-null is the honest reading on both counts. The disease co-occurrence panel beside it is the standing reminder of why the problem is hard in the first place, since the off-diagonal cells show how thoroughly the secondary diseases are entangled with malaria, which is the structural reason a single confident label is the wrong output and a referral under uncertainty is the right one.
""")

# --- 5.8 Evaluation summary --------------------------------------------------
md(r"""## 5.8 Evaluation Summary

The evaluation has tested the model against the three criteria that the triage framing demands, together with a fairness screen, and the results are consistent with one another and with the honesty guardrails set at the outset. On generalisation, the leave-one-site-out comparison showed that dengue detection which appears workable on a pooled sample degrades substantially when the model must transfer to an unseen health district, and a site classifier confirmed why, since the two districts are almost perfectly separable from the clinical record and the dominant axis of symptom variation in the phenotype map tracks geography rather than diagnosis. This forbids presenting the tool as a cross-region detector and motivates the uncertainty-aware framing rather than undermining it. On calibrated uncertainty, per-label conformal prediction produced sets whose empirical coverage matched the nominal target for the two diseases the cohort can support, turning the moderate dengue probability into an explicit and guaranteed statement of when dengue cannot be ruled out. On decision safety, that statement was converted into an asymmetric referral rule that minimises the costly error, a missed dangerous case, at the price of a counted number of confirmatory tests, which is the correct trade for a surge setting. The fairness screen found no gross inequity in the safety-critical quantity, the referral path's sensitivity to true dengue, across age and sex, with the only marked disparity being the district one that simply reflects the real difference in disease prevalence. Taken together these results answer the third research question and satisfy the machine-learning success criterion as it was actually defined in Section 1, which was never raw detection accuracy but safe, honest prioritisation under scarcity. They also mark the boundary of what this cohort can establish. The conformal guarantee is claimed only for malaria and dengue and only within the mixed population, the cross-district result calls for local recalibration before any new deployment, and the fairness screen is a check for gross inequity rather than a powered guarantee given the small subgroups. The prototype interface that would place this referral logic in front of a clinician, and the roadmap toward the multi-region validation these results demand, are the subject of the deployment phase.
""")

md(r"""---

# 6. Deployment

> A prototype triage interface presented as a non-clinical proof of concept, and an honest roadmap toward multi-region generalisation.

The deployment phase asks how the validated model would be placed in front of a user and what would have to be true before it could travel beyond the single region it was trained on. Two things are in scope. The first is the form the artefact takes, a lightweight triage interface that exposes the evaluation logic of Section 5 rather than a raw probability. The second, and the one this section develops in depth because the evaluation made it unavoidable, is an honest account of what generalisation would require, grounded in the cross-cohort evidence already assembled. Consistent with the honesty guardrails stated in Section 1, nothing here is presented as a clinical product; the deliverable is a proof of concept and a roadmap.
""")
md(r"""## 6.1 The Triage Artefact

The intended artefact is deliberately modest in scope and faithful to the evaluation. A frontline health worker records the symptoms present in a febrile patient, optionally adds whatever laboratory values exist, and receives three things rather than a single answer: the per-disease probabilities from the chain-ensemble model of Section 4, the per-label conformal prediction set of Section 5.3 that states which diseases cannot be ruled out at the chosen confidence, and the referral flag of Section 5.4 that fires when dengue remains in the set. The output is therefore the language of triage rather than of diagnosis, which is the only claim the evidence supports: it prioritises who to refer and which confirmatory test would resolve the uncertainty, and it is explicit when it cannot decide. A visible non-clinical disclaimer is part of the interface, not an afterthought, because the cohort size and the single-region origin forbid any stronger reading. The faithfulness criterion that matters for such a tool is that its output match the notebook model exactly, that it remain usable when every laboratory field is blank, which is the operating condition it is designed for, and that it abstain or refer rather than assert on the dangerous and uncertain cases. The interactive realisation of this interface is a natural extension of the analysis already built and is left as the engineering step that follows from it.
""")
md(r"""## 6.2 An Honest Roadmap to Multi-Region Generalisation

The evaluation established two facts that together define the generalisation problem. Within this cohort, Section 5.2 showed that a symptoms-only model trained in one health district transfers poorly to the other because the clinical record carries a strong signature of place. Across cohorts, Sections 4.4.1 and 5.2.1 showed that the qualitative pattern this project confronts, the silent single-label failure on dengue and the moderate-malaria, weak-minority separability of symptoms, recurs in independent Kenyan and Tanzanian paediatric cohorts. The honest conclusion is that the contribution of this work is the formulation, the multi-label and uncertainty-aware reframing with its leakage-controlled missingness analysis, and not a set of transportable weights. A roadmap to deployment in a new region therefore has to treat the model as something to be recalibrated locally rather than shipped intact.

Three commitments follow, and the cross-cohort exercise made each of them concrete rather than rhetorical. The first is local recalibration before use. Because the conformal guarantee of Section 5.3 holds only under exchangeability between calibration and deployment populations, and Section 5.2 showed that assumption breaking across districts within one country, any new facility would need to recalibrate the per-label thresholds on a modest local sample before the coverage statement could be trusted; the leave-one-site-out collapse is the empirical warning that skipping this step would silently void the guarantee. The second is external validation on matched cohorts, of exactly the kind begun in Section 5.2.1, extended from a qualitative convergence check into a prospective evaluation with a shared, harmonised feature definition. Here the cross-cohort work also surfaced the central obstacle to such validation: the Kenyan symptoms were recorded as six grouped indicators rather than as individual signs, so the cohorts could only be compared coarsely and no model could be transferred between them. The practical lesson is that interoperable, granular symptom capture is a precondition for federated evaluation, which is a design requirement for the data collection instrument rather than for the model. The third is a privacy-preserving training strategy. Because health data rarely crosses national borders and should not need to, the appropriate path is federated or few-shot transfer in which the model travels to the data and only model updates, never patient records, are shared. This both respects data sovereignty and addresses the health-data-poverty concern that motivated the project, since it lets a model improve on under-represented populations without extracting their data, in line with the World Health Organization guidance on the ethics and governance of artificial intelligence for health and with the FUTURE-AI principles for trustworthy medical artificial intelligence. Taken together these three commitments describe a tool that is honest about its present limits and specific about what would make it travel, which is the most a single-region proof of concept can responsibly offer.
""")

# ============================================================================
# REFERENCES
# ============================================================================
md(r"""---

## References

- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., and Wirth, R. (2000). *CRISP-DM 1.0: Step-by-Step Data Mining Guide*. CRISP-DM Consortium / SPSS.
- Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., and Müller, K.-R. (2021). Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology. *Machine Learning and Knowledge Extraction*, 3(2), 392-413. https://doi.org/10.3390/make3020020
- Ouedraogo, et al. (2025). *Tabular dataset for AI-based Vector-Borne Disease prediction*. Mendeley Data, V2. DOI 10.17632/cf49v47z4c.2. Licensed CC BY 4.0.
- Mutai, C. K., McSharry, P. E., Ngaruye, I., and Musabanganji, E. (2023). Use of machine learning techniques to identify the predictors of dengue, malaria and typhoid in a Kenyan cohort. *PLOS Global Public Health*. (Single-label dengue detection collapses to zero to 5.5 percent sensitivity.)
- Kotepui, M., et al. (2023). Prevalence and outcomes of malaria and dengue co-infection: a systematic review and meta-analysis in African and Asian populations. *Malaria Journal*. (Co-infection associated with severe-disease odds ratio of approximately 3.94.)
- World Health Organization. (2006). *Communicable diseases following natural disasters: risk assessment and priority interventions*. Geneva: WHO.
- World Health Organization. (2024). *World Malaria Report 2024* and *Dengue and severe dengue, global situation update*. Geneva: WHO.
- Messina, J. P., et al. (2019). The current and future global distribution and population at risk of dengue. *Nature Microbiology*, 4, 1508-1515.
- Kiener, M., LaBeaud, A. D., Ichura, C., Ndenga, B., and Mutuku, F. (2024). *Western and coastal Kenya sick visit data 2019-2022* (dataset). Dryad. https://doi.org/10.5061/dryad.w9ghx3fxc. Licensed CC BY. (Independent paediatric febrile cohort with molecular dengue and rapid-test malaria results, used for external corroboration only.)
- D'Acremont, V., Kilowoko, M., Kyungu, E., Philipina, S., Sangu, W., Kahama-Maro, J., Lengeler, C., Cherpillod, P., Kaiser, L., and Genton, B. (2014). Beyond malaria: causes of fever in outpatient Tanzanian children. *New England Journal of Medicine*, 370, 809-817. (Paediatric Tanzanian fever cohort with blood-smear and rapid-test confirmed diagnoses; dataset licensed CC BY.)
- World Health Organization. (2021). *Ethics and Governance of Artificial Intelligence for Health*. Geneva: WHO (with the 2024 large multi-modal models update).
- Lekadir, K., et al. (2025). FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. *BMJ*, 388, e081554.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = os.path.join(PROJECT_ROOT, "notebook.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")

# Guardrail: assert no em dash or en dash slipped into any markdown cell
for dash, name in [("—", "em dash"), ("–", "en dash")]:
    bad = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and dash in c.source]
    print(f"Cells containing {name} (should be empty):", bad)
