"""
SENTINEL deployment core (frozen engine for the Streamlit triage app).

This module is a FAITHFUL, FROZEN copy of the modelling and evaluation logic that the
graded research notebook builds and validates. It exists so the deployment application can
reproduce, exactly, the same triage behaviour the notebook reports, without the notebook
having to depend on this module (the notebook remains self-contained). Nothing here invents
new methodology: the preprocessing, the ensemble of classifier chains, the per-label
conformal calibration, the referral rule, the value-of-information next-test ranking, and
the short questionnaire are ported verbatim from the notebook, with one structural change
that is asserted to be behaviour-identical (see EnsembleClassifierChains).

Provenance and honesty constraints carried over from the research work:
  * the official Burkina Faso cohort (N=299) is the only data used; no external rows enter;
  * conformal coverage is claimed only for malaria and dengue; yellow fever and typhoid are
    descriptive (too few positives to calibrate), so they never enter a prediction set or
    trigger a referral;
  * the value-of-information next-test ranking and the questionnaire are population-level and
    descriptive, not per-patient claims;
  * this is a research proof of concept on a single-region cohort, NOT a clinical device.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from numpy.random import default_rng

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

# --------------------------------------------------------------------------------------
# Global constants (identical to the notebook)
# --------------------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
ALPHA = 0.10  # target miscoverage: 90% conformal sets should contain the truth

# Disease label order, fixed in Section 3 of the notebook.
TARGET_LABELS = ["malaria", "dengue", "yellow_fever", "typhoid"]
LABELS = ["malaria", "dengue", "typhoid", "yellow_fever"]  # the modelling order (Section 4)
CONF_LABELS = ["malaria", "dengue"]  # guarantee claimed only for these
DISP = {
    "malaria": "Malaria",
    "dengue": "Dengue",
    "yellow_fever": "Yellow fever",
    "typhoid": "Typhoid",
}

# Physiological bounds for the stateless repairs (Section 3.2).
TEMP_LO, TEMP_HI = 34.0, 43.0
PULSE_LO, PULSE_HI = 30.0, 220.0
REFILL_LO, REFILL_HI = 0.0, 10.0

# The twelve signs the Section 5.9 multi-objective (NSGA-II) search selected most consistently,
# snapshotted from one executed run of the notebook. The search is stochastic and its selection
# is deliberately reported as UNSTABLE in the notebook (low cross-resample Jaccard); the honest
# deliverable is therefore "a method for a locally recalibrated questionnaire", and this fixed
# list is one representative example, surfaced so the app can demonstrate the short-form idea
# without re-running the (multi-minute) evolutionary search at app start-up.
QUESTION_COLS_SNAPSHOT = [
    "oliguria",
    "shiver_or_cold_sensation",
    "back_pain_rachiodynia",
    "positive_tourniquet_test",
    "frequent_urination",
    "joint_pain",
    "osteoarthritis",
    "mucosal_skin_pallor_or_anemia",
    "headache_2",
    "chaist_pain",
    "constipation",
    "multiple_convulsions",
]

# Readable clinical labels for the awkward raw column names (Section 6.1 override map).
LABEL_OVERRIDE = {
    "chaist_pain": "Chest pain",  # raw column carries a data-entry typo
    "headache_2": "Headache (cephalalgia field)",  # the second of two headache columns
    "back_pain_rachiodynia": "Back pain (rachialgia)",
    "mucosal_skin_pallor_or_anemia": "Mucosal or skin pallor",
    "shiver_or_cold_sensation": "Shivering or chills",
    "frequent_urination": "Frequent urination",
    "positive_tourniquet_test": "Positive tourniquet test",
    "multiple_convulsions": "Convulsions",
    "joint_pain": "Joint pain",
    "osteoarthritis": "Joint pain (osteoarticular)",
    "oliguria": "Reduced urine output (oliguria)",
    "constipation": "Constipation",
}


def label_for(col: str) -> str:
    """Readable label for a feature column (override map, else a generic prettifier)."""
    if col in LABEL_OVERRIDE:
        return LABEL_OVERRIDE[col]
    return col.replace("_", " ").replace(" oui", "").strip().capitalize()


# --------------------------------------------------------------------------------------
# Stateless text / value repairs (Section 3.1 - 3.3) -- ported verbatim
# --------------------------------------------------------------------------------------
def _fold(s) -> str:
    """Strip accents and upper-case, robust to composed/decomposed Unicode ('Negatif ' -> 'NEGATIF')."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    ).strip().upper()


def _ascii_snake(text, maxlen: int = 32) -> str:
    text = "".join(
        c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c)
    )
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return text[:maxlen] or "feat"


def encode_oui_non(series: pd.Series) -> pd.Series:
    return series.map(
        lambda v: {"OUI": 1.0, "NON": 0.0}.get(_fold(v), np.nan) if pd.notna(v) else np.nan
    ).astype("float")


def encode_pos_neg(series: pd.Series) -> pd.Series:
    return series.map(
        lambda v: {"POSITIF": 1.0, "NEGATIF": 0.0}.get(_fold(v), np.nan) if pd.notna(v) else np.nan
    ).astype("float")


def fix_temperature(val):
    if pd.isna(val):
        return np.nan
    s = re.sub(r"[^0-9.]", "", str(val).strip().replace(",", "."))
    if s in ("", "."):
        return np.nan
    try:
        x = float(s)
    except ValueError:
        return np.nan
    for _ in range(4):
        if x <= TEMP_HI:
            break
        x /= 10.0
    return x if TEMP_LO <= x <= TEMP_HI else np.nan


def fix_pulse(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    if "/" in s:
        s = s.split("/")[0]
    x = pd.to_numeric(s, errors="coerce")
    return float(x) if pd.notna(x) and PULSE_LO <= x <= PULSE_HI else np.nan


def fix_capillary_refill(val):
    if pd.isna(val):
        return np.nan
    x = pd.to_numeric(str(val).replace(",", "."), errors="coerce")
    return float(x) if pd.notna(x) and REFILL_LO <= x <= REFILL_HI else np.nan


def parse_blood_pressure(val):
    if pd.isna(val):
        return (np.nan, np.nan)
    s = str(val).strip().replace("|", "/").replace(" ", "").replace(",", ".")
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            return (np.nan, np.nan)
        sys = pd.to_numeric(parts[0], errors="coerce")
        dia = pd.to_numeric(parts[1], errors="coerce")
        if pd.notna(sys) and pd.notna(dia) and sys < 26 and dia < 26:
            sys, dia = sys * 10, dia * 10
    else:
        digits = re.sub(r"[^0-9]", "", s)
        if len(digits) in (5, 6):
            sys, dia = float(digits[:3]), float(digits[3:])
        elif len(digits) == 4:
            sys, dia = float(digits[:2]), float(digits[2:])
        else:
            v = pd.to_numeric(s, errors="coerce")
            sys, dia = (v, np.nan) if pd.notna(v) else (np.nan, np.nan)
    sys = sys if (pd.notna(sys) and 50 <= sys <= 260) else np.nan
    dia = dia if (pd.notna(dia) and 20 <= dia <= 160) else np.nan
    return (float(sys) if pd.notna(sys) else np.nan, float(dia) if pd.notna(dia) else np.nan)


def fix_age(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", ".")
    if "/" in s:
        num, den = s.split("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return np.nan
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------------------
# Raw loading and the full Section 3 preparation pipeline
# --------------------------------------------------------------------------------------
def load_raw(data_path: str = "data.csv") -> pd.DataFrame:
    """Load the cohort with the verified encoding (utf-8, ';' separator, ',' decimal)."""
    raw = pd.read_csv(data_path, sep=";", decimal=",", encoding="utf-8")
    raw.columns = [c.strip() for c in raw.columns]
    if raw.shape != (300, 109):
        raise ValueError(f"Unexpected raw shape {raw.shape}; expected (300, 109). Check data.csv.")
    return raw


def build_dataset(raw: pd.DataFrame) -> dict:
    """Run the Section 3 preparation pipeline. Returns X, Y, M, site, feature_groups, etc.

    Every step is a stateless, leak-free repair/construction. Stateful transforms (impute,
    scale, rank) are NOT applied here; they are fitted inside folds by build_fold_matrix.
    """
    C = list(raw.columns)
    ID_COLS = [C[0], C[108]]
    COMBINED_DX_COL = C[98]
    SITE_COL = C[0]
    DISEASE_COLS = {
        "malaria": C[99], "dengue": C[100], "chikungunya": C[101], "yellow_fever": C[102],
        "typhoid": C[103], "zika": C[104], "others": C[105], "option_8": C[106],
    }
    LEAK_COL_DENGUE = C[97]
    HOUSEHOLD_DENGUE = C[96]
    LEAK_OR_LABEL_COLS = [LEAK_COL_DENGUE, COMBINED_DX_COL, C[107]]
    disease_set = set(DISEASE_COLS.values())
    HANDLED_EXPLICITLY = {HOUSEHOLD_DENGUE, C[68], C[69]}

    # --- 3.1 selection: classify retained columns by family --------------------------
    binary_symptom_cols, zero_var_cols = [], []
    for col in raw.columns:
        if col in ID_COLS or col in disease_set or col in LEAK_OR_LABEL_COLS or col in HANDLED_EXPLICITLY:
            continue
        if raw[col].nunique(dropna=True) <= 1:
            zero_var_cols.append(col)
            continue
        vals = set(_fold(v) for v in raw[col].dropna().unique())
        if vals <= {"OUI", "NON"}:
            binary_symptom_cols.append(col)

    # --- 3.3 feature construction ----------------------------------------------------
    NAME_OVERRIDES = {
        C[44]: "delirium", C[59]: "moderate_splenomegaly", C[62]: "lymphadenopathy",
        C[70]: "hemoconcentration", C[84]: "alat_asat_elevated", C[85]: "diabetes",
        C[86]: "hypertension", C[87]: "sickle_cell", C[90]: "osteoarthritis",
    }

    def short_handle(col):
        if col in NAME_OVERRIDES:
            return NAME_OVERRIDES[col]
        m = re.search(r"\(([^)]*)\)\s*$", col)
        gloss = _ascii_snake(m.group(1)) if m else ""
        return gloss or _ascii_snake(re.sub(r"\s*\([^)]*\)\s*$", "", col))

    feat = {}
    feature_groups = {
        "symptom": [], "vital": [], "lab": [], "lab_ambiguous": [], "demographic": [],
        "confirmatory_test": [], "history": [], "derived": [],
    }

    def add(name, series, group):
        h, i = name, 2
        while h in feat:
            h = f"{name}_{i}"
            i += 1
        feat[h] = series
        feature_groups[group].append(h)
        return h

    AGE_COL, WEIGHT_COL, GENDER_COL = C[2], C[3], C[1]
    TEMP_COL, PULSE_COL, BP_COL = C[63], C[65], C[66]
    HISTORY_BIN = {C[85], C[86], C[87], C[90], C[93], C[95]}
    DUP_URINATION = {C[35], C[38]}

    for col in binary_symptom_cols:
        if col in DUP_URINATION:
            continue
        add(short_handle(col), encode_oui_non(raw[col]),
            "history" if col in HISTORY_BIN else "symptom")

    _u1, _u2 = encode_oui_non(raw[C[35]]), encode_oui_non(raw[C[38]])
    add("frequent_urination",
        (_u1.fillna(0) + _u2.fillna(0)).clip(0, 1).where(~(_u1.isna() & _u2.isna())), "symptom")
    add("household_dengue", encode_oui_non(raw[HOUSEHOLD_DENGUE]), "history")

    add("tdr_positive", encode_pos_neg(raw[C[68]]), "confirmatory_test")
    add("thick_smear_positive", encode_pos_neg(raw[C[69]]), "confirmatory_test")

    add("age_years", raw[AGE_COL].apply(fix_age), "demographic")
    add("is_female", raw[GENDER_COL].map(
        lambda v: {"FEMME": 1.0, "HOMME": 0.0}.get(_fold(v), np.nan) if pd.notna(v) else np.nan),
        "demographic")
    add("weight_kg", pd.to_numeric(raw[WEIGHT_COL], errors="coerce"), "demographic")
    _mua = raw[C[4]].map(lambda v: pd.to_numeric(str(v).replace(",", "."), errors="coerce"))
    add("mua_cm", _mua.where(_mua.between(5, 40)), "demographic")

    add("temp_c", raw[TEMP_COL].apply(fix_temperature), "vital")
    _rr = pd.to_numeric(raw[C[64]].astype(str).str.replace(",", "."), errors="coerce")
    add("resp_rate", _rr.where(_rr.between(8, 80)), "vital")
    add("pulse_bpm", raw[PULSE_COL].apply(fix_pulse), "vital")
    _bp = raw[BP_COL].apply(parse_blood_pressure)
    add("bp_systolic", _bp.apply(lambda t: t[0]), "vital")
    add("bp_diastolic", _bp.apply(lambda t: t[1]), "vital")
    add("capillary_refill_s", raw[C[67]].apply(fix_capillary_refill), "vital")
    add("fever_type_recurrent", raw[C[8]].map(
        lambda v: {"RECURRENTE": 1.0, "INTERMITTENTE": 0.0}.get(_fold(v), np.nan)
        if pd.notna(v) else np.nan), "vital")

    _plt = pd.to_numeric(raw[C[80]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    add("platelets", _plt.mask(_plt < 1000, _plt * 1000), "lab")

    for name, col in {"hematocrit": C[73], "wbc": C[79], "neutrophils": C[81],
                      "lymphocytes": C[77], "creatinine_elev": C[83]}.items():
        add(name, pd.to_numeric(raw[col].astype(str).str.replace(",", ".", regex=False),
                                errors="coerce"), "lab_ambiguous")

    _sym_so_far = list(feature_groups["symptom"])
    add("symptom_burden",
        pd.DataFrame({h: feat[h] for h in _sym_so_far}).sum(axis=1, min_count=1), "derived")
    _age = raw[AGE_COL].apply(fix_age)
    _band = pd.cut(_age, [0, 5, 15, 200], labels=["U5", "5-15", "15+"])
    add("age_band", _band.cat.codes.where(_band.notna(), np.nan).astype("float"), "derived")

    clean = pd.DataFrame(feat, index=raw.index)

    # --- 3.4 target and cohort finalisation ------------------------------------------
    Y_all = pd.DataFrame(
        {lbl: pd.to_numeric(raw[DISEASE_COLS[lbl]], errors="coerce") for lbl in TARGET_LABELS},
        index=raw.index)
    others = pd.to_numeric(raw[DISEASE_COLS["others"]], errors="coerce").rename("others")
    all_nan_label = Y_all.isna().all(axis=1)
    keep = ~all_nan_label
    X = clean.loc[keep].reset_index(drop=True)
    Y = Y_all.loc[keep].astype(int).reset_index(drop=True)
    others = others.loc[keep].reset_index(drop=True)

    # --- 3.5 site integration --------------------------------------------------------
    raw299 = raw.drop(index=int(raw.index[all_nan_label][0])).reset_index(drop=True)
    site = raw299[SITE_COL].str.strip().map(
        {"CMA de DO": "DO", "CMA de DAFRA": "DAFRA"}).rename("site")

    # --- 3.6 missingness mask --------------------------------------------------------
    M = X.isna().astype(int).add_suffix("__isna")
    CONF_FEATS = list(feature_groups["confirmatory_test"])
    mask_confirmatory = [f"{f}__isna" for f in CONF_FEATS if f"{f}__isna" in M.columns]
    mask_generic = [c for c in M.columns if c not in mask_confirmatory]

    return dict(
        X=X, Y=Y, M=M, site=site, others=others, feature_groups=feature_groups,
        mask_generic=mask_generic, mask_confirmatory=mask_confirmatory,
    )


# --------------------------------------------------------------------------------------
# Feature regimes and the leak-free fold pipeline (Section 4.2) -- ported verbatim
# --------------------------------------------------------------------------------------
def make_regimes(data: dict):
    """Build the regime column specs and the in-fold transform column lists."""
    X = data["X"]
    fg = data["feature_groups"]
    COLS_SYMPTOM = fg["symptom"] + fg["history"] + fg["demographic"] + fg["derived"]
    COLS_LAB = fg["vital"] + fg["lab"] + fg["lab_ambiguous"] + fg["confirmatory_test"]
    RANK_COLS = fg["lab_ambiguous"]
    SCALE_COLS = [c for c in fg["vital"] if c in X.columns]
    IMPUTE_COLS = [c for c in COLS_LAB if c in X.columns]
    REGIMES = {
        "symptoms": dict(base=COLS_SYMPTOM, mask=[], labs=False),
        "symptoms+mask": dict(base=COLS_SYMPTOM,
                              mask=list(data["mask_generic"]) + list(data["mask_confirmatory"]),
                              labs=False),
        "symptoms+labs": dict(base=COLS_SYMPTOM, mask=[], labs=True),
    }
    return dict(REGIMES=REGIMES, COLS_SYMPTOM=COLS_SYMPTOM, COLS_LAB=COLS_LAB,
                RANK_COLS=RANK_COLS, SCALE_COLS=SCALE_COLS, IMPUTE_COLS=IMPUTE_COLS)


def build_fold_matrix(spec, tr_idx, te_idx, X, M, regimes):
    """Return (Xtr, Xte) for a regime, fitting all stateful transforms on train rows only."""
    IMPUTE_COLS = regimes["IMPUTE_COLS"]
    RANK_COLS = regimes["RANK_COLS"]
    SCALE_COLS = regimes["SCALE_COLS"]
    base_cols = [c for c in spec["base"] if c in X.columns]
    Xtr = X.iloc[tr_idx][base_cols].copy()
    Xte = X.iloc[te_idx][base_cols].copy()

    if spec["labs"]:
        lab_tr = X.iloc[tr_idx][IMPUTE_COLS].copy()
        lab_te = X.iloc[te_idx][IMPUTE_COLS].copy()
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


# --------------------------------------------------------------------------------------
# Models (Section 4.4) -- fit_ecc is the reference; EnsembleClassifierChains is a
# behaviour-identical stateful refactor used by the app for fast repeated scoring.
# --------------------------------------------------------------------------------------
def base_learner():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=1)


def fit_ecc(Xtr, Ytr, Xte, n_chains=8, rng=None):
    """REFERENCE ensemble of classifier chains (ported verbatim from the notebook).

    Kept so export_model.py can assert that EnsembleClassifierChains.predict_proba reproduces
    this function exactly. The app itself does not call this at inference time."""
    rng = rng or default_rng(SEED)
    nL = len(LABELS)
    acc = np.zeros((len(Xte), nL))
    m_idx = LABELS.index("malaria")
    for _ in range(n_chains):
        rest = [j for j in range(nL) if j != m_idx]
        rng.shuffle(rest)
        order = [m_idx] + rest
        Xtr_aug = Xtr.copy()
        Xte_aug = Xte.copy()
        P_chain = np.zeros((len(Xte), nL))
        for pos, j in enumerate(order):
            yj = Ytr[:, j]
            if yj.sum() == 0:
                P_chain[:, j] = 0.0
                feat_te = np.zeros(len(Xte))
            else:
                clf = base_learner().fit(Xtr_aug, yj)
                feat_te = clf.predict_proba(Xte_aug)[:, 1]
                P_chain[:, j] = feat_te
            Xtr_aug[f"_chain_{j}"] = yj
            Xte_aug[f"_chain_{j}"] = feat_te
        acc += P_chain
    return acc / n_chains


class EnsembleClassifierChains:
    """Stateful refactor of fit_ecc: fit once on the full cohort, then score new patients fast.

    Fit reproduces fit_ecc's training loop exactly (same seeded chain orders, true prior labels
    cascaded forward on train). predict_proba reproduces fit_ecc's TEST path exactly (predicted
    prior probabilities cascaded forward). With identical chain orders and the deterministic
    LightGBM base learner, ecc.predict_proba(X) == fit_ecc(X, Y, X); export_model.py asserts it.
    """

    def __init__(self, n_chains: int = 8, seed: int = SEED):
        self.n_chains = n_chains
        self.seed = seed
        self.chains_ = []  # list of (order, [model_or_None per position])
        self.columns_ = None

    def fit(self, Xtr: pd.DataFrame, Ytr: np.ndarray):
        Xtr = Xtr.copy()
        self.columns_ = list(Xtr.columns)
        rng = default_rng(self.seed)
        nL = len(LABELS)
        m_idx = LABELS.index("malaria")
        self.chains_ = []
        for _ in range(self.n_chains):
            rest = [j for j in range(nL) if j != m_idx]
            rng.shuffle(rest)
            order = [m_idx] + rest
            Xtr_aug = Xtr.copy()
            models = []
            for j in order:
                yj = Ytr[:, j]
                if yj.sum() == 0:
                    models.append(None)
                    feat_tr = np.zeros(len(Xtr))
                else:
                    clf = base_learner().fit(Xtr_aug, yj)
                    models.append(clf)
                    feat_tr = yj  # true label cascaded forward on train (matches fit_ecc)
                Xtr_aug[f"_chain_{j}"] = feat_tr
            self.chains_.append((order, models))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns_ is not None:
            X = X.reindex(columns=self.columns_, fill_value=0.0)
        nL = len(LABELS)
        acc = np.zeros((len(X), nL))
        for order, models in self.chains_:
            X_aug = X.copy()
            P_chain = np.zeros((len(X), nL))
            for pos, j in enumerate(order):
                clf = models[pos]
                if clf is None:
                    feat = np.zeros(len(X))
                else:
                    feat = clf.predict_proba(X_aug)[:, 1]
                P_chain[:, j] = feat
                X_aug[f"_chain_{j}"] = feat
            acc += P_chain
        return acc / self.n_chains


def fit_binary_relevance_models(Xtr: pd.DataFrame, Ytr: np.ndarray) -> dict:
    """One independent classifier per label (used only for the SHAP explanation panel).

    These are NOT the chain ensemble that drives the probabilities, prediction sets, and
    referral; they are a per-disease model over the same signs, used to attribute which
    symptoms most push a given disease's risk up or down in a way that is straightforward to
    explain. The app states this distinction explicitly.
    """
    models = {}
    for j, lbl in enumerate(LABELS):
        yj = Ytr[:, j]
        if yj.sum() == 0:
            models[lbl] = None
            continue
        models[lbl] = base_learner().fit(Xtr, yj)
    return models


def strat_splits(Yv):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros((len(Yv), 1)), Yv))


def site_splits(Yv, groups_site):
    cv = GroupKFold(n_splits=len(np.unique(groups_site)))
    return list(cv.split(np.zeros((len(Yv), 1)), groups=groups_site))


def oof_probabilities(fit_fn, spec, splits, X, M, regimes, Yv):
    P = np.zeros((len(Yv), len(LABELS)))
    for tr, te in splits:
        Xtr, Xte = build_fold_matrix(spec, tr, te, X, M, regimes)
        P[te] = fit_fn(Xtr, Yv[tr], Xte)
    return P


# --------------------------------------------------------------------------------------
# Per-label conformal calibration (Section 5.3) -- ported verbatim
# --------------------------------------------------------------------------------------
def conformal_label(P, j, Yv, cal_idx, test_idx, alpha=ALPHA):
    s = 1.0 - P[:, j]
    cal_pos = cal_idx[Yv[cal_idx, j] == 1]
    if len(cal_pos) < 10:
        return None
    qhat = np.quantile(s[cal_pos], 1 - alpha, method="higher")
    in_set = (s[test_idx] <= qhat)
    test_pos = Yv[test_idx, j] == 1
    cov = in_set[test_pos].mean()
    return dict(qhat=float(qhat), in_set=in_set, coverage=float(cov),
                set_rate=float(in_set.mean()), n_cal_pos=int(len(cal_pos)),
                n_test_pos=int(test_pos.sum()))


def wilson_ci(k, nobs, z=1.96):
    if nobs == 0:
        return (np.nan, np.nan)
    p = k / nobs
    d = 1 + z**2 / nobs
    centre = (p + z**2 / (2 * nobs)) / d
    half = (z * np.sqrt(p * (1 - p) / nobs + z**2 / (4 * nobs**2))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibrate_conformal(P_ecc, Yv):
    """Reproduce the Section 5.3 calibration: one seeded 50/50 split, per-label qhat for
    malaria and dengue. Returns the conf dict and the cal/test indices."""
    rng_c = default_rng(SEED)
    n = len(Yv)
    perm = rng_c.permutation(n)
    cal_idx, test_idx = perm[: n // 2], perm[n // 2:]
    conf = {}
    for lbl in CONF_LABELS:
        conf[lbl] = conformal_label(P_ecc, LABELS.index(lbl), Yv, cal_idx, test_idx)
    return conf, cal_idx, test_idx


def referral_accounting(conf, P_ecc, Yv, test_idx):
    """Section 5.4 referral accounting on the conformal test split (for the export asserts)."""
    di_, mi_ = LABELS.index("dengue"), LABELS.index("malaria")
    in_dengue = conf["dengue"]["in_set"]
    y_test = Yv[test_idx]
    true_dengue = y_test[:, di_] == 1
    refer = in_dengue
    caught = int((refer & true_dengue).sum())
    missed = int((~refer & true_dengue).sum())
    overref = int((refer & ~true_dengue).sum())
    naive_refer = (P_ecc[test_idx, di_] >= 0.5)
    naive_caught = int((naive_refer & true_dengue).sum())
    return dict(referral_rate=float(refer.mean()), caught=caught, missed=missed,
                overref=overref, dengue_n=int(true_dengue.sum()),
                referred=int(refer.sum()), naive_referred=int(naive_refer.sum()),
                naive_caught=naive_caught)


# --------------------------------------------------------------------------------------
# Local recalibration demonstration (Section 6.2 / 6.3)
#
# The conformal coverage guarantee of Section 5.3 holds only under exchangeability between the
# calibration and deployment populations, and Section 5.2 showed that assumption breaking across
# health districts. These helpers make that concrete: they recalibrate the per-label threshold on
# a chosen "local" population (a health district, as a within-cohort proxy for a new facility) and
# measure the coverage it achieves on each district. The contrast between applying the pooled
# threshold to a single district and recalibrating on that district shows why a deployed model has
# to be recalibrated locally rather than shipped with fixed thresholds. Every number reuses the
# same nonconformity score and conformal quantile as Section 5.3; only the calibration and
# evaluation row masks change.
# --------------------------------------------------------------------------------------
def _pop_mask(site, pop):
    """Boolean mask for a population: 'pooled' (everyone) or a specific site value."""
    site = np.asarray(site)
    if pop == "pooled":
        return np.ones(len(site), dtype=bool)
    return site == pop


def recalibrate_coverage(P_ecc, Yv, site, label, cal_pop, eval_pop, alpha=ALPHA):
    """Recalibrate the per-label threshold on cal_pop and measure coverage on eval_pop.

    Returns qhat, empirical coverage on eval_pop's true positives (with a Wilson interval), the
    number of evaluation positives, and the set rate (the fraction of eval patients kept in the
    set, which for dengue is the referral rate). Returns None when cal_pop has fewer than ten
    positives of the label, the same minimum the Section 5.3 calibration requires.
    """
    j = LABELS.index(label)
    s = 1.0 - P_ecc[:, j]
    cal_mask = _pop_mask(site, cal_pop)
    eval_mask = _pop_mask(site, eval_pop)
    cal_pos = np.where(cal_mask & (Yv[:, j] == 1))[0]
    if len(cal_pos) < 10:
        return None
    qhat = float(np.quantile(s[cal_pos], 1 - alpha, method="higher"))
    ev = np.where(eval_mask)[0]
    in_set = s[ev] <= qhat
    ev_pos = Yv[ev, j] == 1
    n_ev_pos = int(ev_pos.sum())
    cov = float(in_set[ev_pos].mean()) if n_ev_pos else float("nan")
    lo, hi = wilson_ci(int(round(cov * n_ev_pos)), n_ev_pos) if n_ev_pos else (float("nan"), float("nan"))
    return dict(qhat=qhat, coverage=cov, ci_lo=lo, ci_hi=hi, n_eval_pos=n_ev_pos,
                n_cal_pos=int(len(cal_pos)), set_rate=float(in_set.mean()),
                p_min=1.0 - qhat)


def recalibration_matrix(recal_data, labels=None, alpha=ALPHA):
    """Full calibration-source by evaluation-site matrix for the local-recalibration demo.

    Calibration sources are the pooled cohort plus each individual site; evaluation populations
    are each site. Returns a nested dict: out[label][cal_pop][eval_site] = recalibrate_coverage(...).
    """
    if labels is None:
        labels = list(CONF_LABELS)
    P_ecc = recal_data["P_ecc"]
    Yv = recal_data["Yv"]
    site = np.asarray(recal_data["site"])
    sites = sorted(np.unique(site).tolist())
    cal_sources = ["pooled"] + sites
    out = {}
    for lbl in labels:
        out[lbl] = {}
        for cal_pop in cal_sources:
            out[lbl][cal_pop] = {}
            for eval_site in sites:
                out[lbl][cal_pop][eval_site] = recalibrate_coverage(
                    P_ecc, Yv, site, lbl, cal_pop, eval_site, alpha=alpha)
    return dict(matrix=out, sites=sites, cal_sources=cal_sources, alpha=alpha,
                n_positives={lbl: {s: int(Yv[site == s, LABELS.index(lbl)].sum()) for s in sites}
                             for lbl in labels})


# --------------------------------------------------------------------------------------
# Value of information: which test to recommend next (Section 5.5) -- ported verbatim
# --------------------------------------------------------------------------------------
def _entropy(counts):
    c = np.asarray(counts, float)
    c = c[c > 0]
    c = c / c.sum()
    return float(-(c * np.log2(c)).sum())


def _mi(x, y):
    ct = pd.crosstab(x, y).values.astype(float)
    if ct.sum() == 0:
        return 0.0
    return max(0.0, _entropy(ct.sum(1)) + _entropy(ct.sum(0)) - _entropy(ct.flatten()))


def _cmi(x, y, z):
    out = 0.0
    for zv in np.unique(z):
        m = (z == zv)
        if m.sum() > 2:
            out += m.mean() * _mi(x[m], y[m])
    return out


def _voi_cost(name):
    if "tdr" in name or "thick_smear" in name:
        return 1.0
    return 3.0


def _voi_signal(series):
    s = series
    if s.dropna().nunique() <= 1:
        return None
    if set(pd.unique(s.dropna())) <= {0.0, 1.0}:
        v = s.fillna(0).astype(int)
    else:
        med = s.median()
        v = (s >= med).astype(int)
        v[s.isna()] = 0
    return pd.Series(v.values, name=series.name)


def compute_voi(X, Yv, feature_groups):
    prof_codes, _ = pd.factorize(pd.Index([tuple(r) for r in Yv.astype(int)], dtype=object))
    prof = pd.Series(prof_codes, name="profile")
    mal_v = Yv[:, LABELS.index("malaria")].astype(int)
    den_v = Yv[:, LABELS.index("dengue")].astype(int)
    candidates = {}
    for f in feature_groups["confirmatory_test"]:
        if f in X.columns:
            candidates[f] = X[f]
    for f in (feature_groups["lab"] + feature_groups["lab_ambiguous"]):
        if f in X.columns:
            candidates[f] = X[f]
    rows = []
    for name, series in candidates.items():
        sig = _voi_signal(series)
        if sig is None:
            continue
        info_profile = _mi(sig, prof)
        info_dengue_adj = _cmi(sig, pd.Series(den_v), mal_v)
        rows.append(dict(test=name, info_profile=info_profile, info_dengue_adj=info_dengue_adj,
                         cost=_voi_cost(name), n_observed=int(series.notna().sum()),
                         info_per_cost=info_profile / _voi_cost(name)))
    voi = pd.DataFrame(rows).sort_values("info_profile", ascending=False).reset_index(drop=True)
    return voi


def pretty_test(name):
    return {"tdr_positive": "malaria rapid test (TDR)",
            "thick_smear_positive": "blood smear (thick film)"}.get(
        name, name.replace("_", " ").replace(" elev", "").strip())


# --------------------------------------------------------------------------------------
# Triage decision (Section 6.1) -- the function the app calls per patient
# --------------------------------------------------------------------------------------
def triage(probs, qhat, next_test):
    """Map per-label probabilities to the triage decision (prediction set + referral).

    probs: dict label -> probability (from the chain ensemble).
    qhat:  dict label -> conformal threshold (None for uncalibrated labels).
    Returns prob, prediction set, per-label status, inclusion thresholds, referral, next test.
    """
    pred_set, status, p_min = [], {}, {}
    for lbl in LABELS:
        q = qhat.get(lbl)
        if q is None:
            status[lbl] = "uncalibrated"
            p_min[lbl] = None
            continue
        in_set = (1.0 - probs[lbl]) <= q
        status[lbl] = "in set" if in_set else "ruled out"
        p_min[lbl] = 1.0 - q
        if in_set:
            pred_set.append(lbl)
    refer = "dengue" in pred_set
    return dict(prob=dict(probs), pred_set=pred_set, status=status, p_min=p_min,
                refer=refer, next_test=next_test if refer else None)


def score_symptoms(symptoms_present, ecc, deploy_cols):
    """Score a hypothetical new patient from a set of present-symptom column names.

    Lab-free operating condition: marked symptoms = 1, everything else (including every
    laboratory field) left at 0 / absent. Returns a dict label -> probability."""
    row = pd.DataFrame(0.0, index=[0], columns=deploy_cols)
    for c in symptoms_present:
        if c in row.columns:
            row.loc[0, c] = 1.0
    probs = ecc.predict_proba(row)[0]
    return {LABELS[j]: float(probs[j]) for j in range(len(LABELS))}


def score_record(record, ecc, deploy_cols):
    """Score a patient from a full feature record (symptoms + any provided lab/vital values).

    record: dict column -> value (missing columns default to 0/absent, the lab-free default).
    Used when the clinician optionally enters laboratory or vital measurements."""
    row = pd.DataFrame(0.0, index=[0], columns=deploy_cols)
    for c, v in record.items():
        if c in row.columns and v is not None and not (isinstance(v, float) and np.isnan(v)):
            row.loc[0, c] = float(v)
    probs = ecc.predict_proba(row)[0]
    return {LABELS[j]: float(probs[j]) for j in range(len(LABELS))}


def shap_top_symptoms(label, record, br_models, deploy_cols, feature_groups, top_k=6):
    """Top contributing signs for one disease, via SHAP on the per-label model.

    Honest framing (stated in the UI): this attributes the per-disease binary-relevance model,
    a model over the same signs, NOT the chain ensemble that produces the headline probability.
    Returns a list of (readable_label, signed_contribution) sorted by magnitude.
    """
    import shap

    model = br_models.get(label)
    if model is None:
        return []
    row = pd.DataFrame(0.0, index=[0], columns=deploy_cols)
    for c, v in record.items():
        if c in row.columns and v is not None and not (isinstance(v, float) and np.isnan(v)):
            row.loc[0, c] = float(v)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(row)
    if isinstance(sv, list):  # older SHAP returns [class0, class1]
        sv = sv[1]
    vals = np.asarray(sv).reshape(-1)
    non_symptom = set(feature_groups.get("demographic", []) + feature_groups.get("derived", []))
    pairs = [(deploy_cols[i], float(vals[i])) for i in range(len(deploy_cols))
             if deploy_cols[i] not in non_symptom and abs(vals[i]) > 1e-9]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return [(label_for(c), v) for c, v in pairs[:top_k]]


# --------------------------------------------------------------------------------------
# Orchestration: build the full deployment engine from the raw cohort
# --------------------------------------------------------------------------------------
def build_engine(data_path: str = "data.csv") -> dict:
    """Build everything the app needs, faithful to the notebook, from the raw cohort.

    Used by export_model.py (to freeze a model.joblib) and by the app's retrain fallback.
    """
    raw = load_raw(data_path)
    data = build_dataset(raw)
    X, Y, M, site = data["X"], data["Y"], data["M"], data["site"]
    feature_groups = data["feature_groups"]
    regimes = make_regimes(data)
    Yv = Y[LABELS].values.astype(int)
    groups_site = site.values

    # Deployment feature matrix = the symptoms-only regime over the full cohort (stateless).
    spec_sym = regimes["REGIMES"]["symptoms"]
    _all = np.arange(len(Yv))
    X_deploy, _ = build_fold_matrix(spec_sym, _all, _all, X, M, regimes)
    deploy_cols = list(X_deploy.columns)

    # Shippable chain ensemble, fit on the full cohort (the single model a deployment ships).
    ecc = EnsembleClassifierChains(n_chains=8, seed=SEED).fit(X_deploy, Yv)

    # Per-label models for the SHAP explanation panel (explanation only; not the headline model).
    br_models = fit_binary_relevance_models(X_deploy, Yv)

    # Pooled out-of-fold chain probabilities, then the Section 5.3 conformal calibration.
    splits_s = strat_splits(Yv)
    P_ecc = oof_probabilities(fit_ecc, spec_sym, splits_s, X, M, regimes, Yv)
    conf, cal_idx, test_idx = calibrate_conformal(P_ecc, Yv)
    qhat = {l: (conf[l]["qhat"] if l in CONF_LABELS and conf.get(l) is not None else None)
            for l in LABELS}

    # Value-of-information next-test recommendation (Section 5.5).
    voi = compute_voi(X, Yv, feature_groups)
    voi_den = voi.sort_values("info_dengue_adj", ascending=False)
    next_test = pretty_test(voi_den["test"].iloc[0]) if len(voi_den) else "a confirmatory panel"

    # The short questionnaire (Section 5.9 snapshot), restricted to columns actually present.
    question_cols = [c for c in QUESTION_COLS_SNAPSHOT if c in deploy_cols]

    # The small artifact for the local-recalibration demo (Section 6.2): out-of-fold probabilities,
    # labels, and site. Tiny (a few KB); P_ecc stored as float32 to keep the payload small.
    recal_data = dict(P_ecc=P_ecc.astype(np.float32), Yv=Yv.astype(np.int8),
                      site=list(site.values), labels=list(LABELS))

    return dict(
        ecc=ecc,
        br_models=br_models,
        qhat=qhat,
        conf=conf,
        next_test=next_test,
        deploy_cols=deploy_cols,
        question_cols=question_cols,
        feature_groups=feature_groups,
        alpha=ALPHA,
        labels=LABELS,
        disp=DISP,
        site=site,
        recal_data=recal_data,
        # quantities retained so export_model.py can assert faithfulness to the notebook
        _P_ecc=P_ecc,
        _Yv=Yv,
        _X_deploy=X_deploy,
        _test_idx=test_idx,
        _voi=voi,
    )
