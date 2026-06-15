"""
SENTINEL - uncertainty-aware triage and referral assistant (deployment prototype).

A Streamlit interface over the frozen research engine in sentinel_core.py. It turns a clinical
sign checklist into the four triage outputs the research notebook defines and validates:
  1. a calibrated per-disease probability (ensemble of classifier chains, Section 4);
  2. a per-label conformal prediction set: the diseases that cannot be ruled out at 90% (5.3);
  3. an asymmetric referral flag that fires when dengue cannot be ruled out (5.4);
  4. the value-of-information next test that most resolves the residual uncertainty (5.5).

The model is symptoms-only by design (it must run when every laboratory field is blank, which is
the surge condition it is built for), so entering a laboratory result does not change the score;
instead the assistant tells you which test would help most. Conformal guarantees are claimed only
for malaria and dengue; yellow fever and typhoid are shown for context and never drive a referral.

NOT FOR CLINICAL USE - a research proof of concept on a single-region cohort of 299 patients.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

import sentinel_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
META_PATH = os.path.join(HERE, "meta.json")

# Disease colours, matching the research notebook / report palette.
BAR_COLOR = {"malaria": "#d1495b", "dengue": "#3a6ea5", "typhoid": "#2e8b57", "yellow_fever": "#e0a01e"}

# --------------------------------------------------------------------------------------
# Bilingual UI strings (the cohort is francophone West Africa, so French is offered).
# Clinical sign labels are translated for the short questionnaire; the rest of the chrome
# is fully bilingual.
# --------------------------------------------------------------------------------------
TEXT = {
    "en": {
        "tagline": "Uncertainty-aware triage and referral assistant for co-circulating "
                   "vector-borne febrile disease, built for the surge that follows floods and "
                   "other disasters, when confirmatory testing is scarcest.",
        "disclaimer": "NOT FOR CLINICAL USE - research proof of concept, single-region cohort "
                      "(N=299). It supports the prioritisation of scarce confirmatory testing; "
                      "it does not diagnose and does not replace clinical judgement.",
        "sidebar_settings": "Settings",
        "language": "Language",
        "form_mode": "Symptom entry",
        "short_form": "Short form (12-sign questionnaire)",
        "full_form": "Full symptom list",
        "short_help": "The twelve signs an evolutionary search (Section 5.9) selected most "
                      "consistently as a compact bedside questionnaire.",
        "demographics": "Patient (optional)",
        "age": "Age (years)",
        "sex": "Sex",
        "sex_unknown": "Not specified",
        "sex_female": "Female",
        "sex_male": "Male",
        "presets": "Illustrative examples",
        "preset_help": "Hypothetical presentations to demonstrate the behaviour; not real patients.",
        "preset_none": "(clear)",
        "preset_malaria": "Malaria-like presentation",
        "preset_dengue": "Dengue-suspicious presentation",
        "signs_present": "Clinical signs present",
        "signs_caption": "Tick the signs the patient presents. The triage updates live.",
        "low_prob_note": "Probabilities are low when only a few signs are ticked: this is by "
                         "design, since a real patient presents many signs at once. The "
                         "prediction set and referral remain meaningful.",
        "result": "Triage decision",
        "refer": "REFER / ESCALATE",
        "monitor": "Monitor - no referral trigger",
        "next_test": "Recommended next test",
        "calibrated": "Calibrated decision (malaria, dengue)",
        "context_only": "Shown for context only (not calibrated)",
        "context_note": "Yellow fever and typhoid have too few cases here (12 and 29) to "
                        "calibrate a reliable threshold, so they never enter the prediction set "
                        "or trigger a referral; their probabilities are informational.",
        "pred_set": "Prediction set (cannot be ruled out at {conf}% confidence)",
        "in_set": "IN SET",
        "ruled_out": "ruled out",
        "kept_if": "kept if &#8805;{thr}%",
        "needs": "needs &#8805;{thr}%",
        "empty_set": "empty",
        "why": "Why this assessment",
        "why_help": "Top signs pushing each calibrated disease up (+) or down (-), attributed to a "
                    "per-disease model over the same signs. This explains the direction of the "
                    "evidence; the headline probability is from the chain ensemble.",
        "why_increase": "increases",
        "why_decrease": "decreases",
        "why_none": "No marked sign moves this disease appreciably.",
        "equity": "Equity note",
        "equity_text": "A fairness audit (Section 5.6) found the referral rule fires at a similar "
                       "sensitivity across sex and the better-powered age bands, but the under-five "
                       "group is too small to certify. Read subgroup behaviour with that caveat.",
        "recal_title": "Generalisation: why this model must be recalibrated locally",
        "recal_intro": "The 90% conformal coverage guarantee holds only when the calibration and "
                       "deployment populations match. The cohort spans two health districts whose "
                       "clinical records differ sharply, so the table below treats each district as "
                       "a stand-in for a new facility: it shows the coverage the threshold achieves "
                       "when calibrated on the pooled cohort, or recalibrated on each district, and "
                       "applied to each district. It is the interactive form of the roadmap in "
                       "Section 6.3.",
        "recal_dengue_h": "Dengue coverage (target 90%)",
        "recal_malaria_h": "Malaria coverage (target 90%)",
        "recal_calon": "Threshold calibrated on",
        "recal_at": "Coverage at",
        "recal_pooled": "Pooled cohort (shipped)",
        "recal_local": "{site} (local recalibration)",
        "recal_takeaway": "The shipped (pooled) threshold covers dengue at only {cov_pooled:.0f}% in "
                          "the dengue-scarce district {scarce}, well below the 90% target, because "
                          "its score distribution differs. Recalibrating on local {scarce} data "
                          "restores coverage to {cov_local:.0f}%, at the cost of a higher referral "
                          "rate. Malaria, the common disease, transfers well and stays near 90% "
                          "regardless. This is why a deployed model travels by local recalibration, "
                          "not by shipping fixed thresholds; note that local {scarce} dengue rests "
                          "on only {n_scarce} cases, so granular local data is itself a precondition.",
        "recal_note": "Cells where a threshold is evaluated on the same district it was calibrated "
                      "on are the within-population target and read slightly optimistically; the "
                      "pooled and cross-district cells are the genuine transfer numbers. Coverage is "
                      "shown with the evaluation positive count, since the dengue cells rest on few "
                      "cases.",
        "how_title": "How this works",
        "card_title": "Model card and provenance",
        "no_signs": "No signs marked yet. Tick the signs the patient presents to see the triage.",
        "retrain_note": "Frozen model not found; rebuilt from the cohort in-session. Run "
                        "export_model.py to cache it.",
    },
    "fr": {
        "tagline": "Assistant de triage et d'orientation tenant compte de l'incertitude pour les "
                   "maladies febriles a transmission vectorielle co-circulantes, concu pour la "
                   "vague qui suit les inondations et autres catastrophes, quand les tests de "
                   "confirmation sont les plus rares.",
        "disclaimer": "NE PAS UTILISER EN CLINIQUE - preuve de concept de recherche, cohorte d'une "
                      "seule region (N=299). L'outil aide a prioriser les tests de confirmation "
                      "rares ; il ne pose pas de diagnostic et ne remplace pas le jugement clinique.",
        "sidebar_settings": "Parametres",
        "language": "Langue",
        "form_mode": "Saisie des symptomes",
        "short_form": "Formulaire court (questionnaire de 12 signes)",
        "full_form": "Liste complete des symptomes",
        "short_help": "Les douze signes qu'une recherche evolutionnaire (Section 5.9) a selectionnes "
                      "le plus systematiquement comme questionnaire de chevet compact.",
        "demographics": "Patient (optionnel)",
        "age": "Age (annees)",
        "sex": "Sexe",
        "sex_unknown": "Non precise",
        "sex_female": "Feminin",
        "sex_male": "Masculin",
        "presets": "Exemples illustratifs",
        "preset_help": "Presentations hypothetiques pour illustrer le comportement ; pas de vrais "
                       "patients.",
        "preset_none": "(effacer)",
        "preset_malaria": "Presentation evocatrice de paludisme",
        "preset_dengue": "Presentation suspecte de dengue",
        "signs_present": "Signes cliniques presents",
        "signs_caption": "Cochez les signes presentes par le patient. Le triage se met a jour en direct.",
        "low_prob_note": "Les probabilites sont faibles quand peu de signes sont coches : c'est "
                         "voulu, car un vrai patient presente de nombreux signes a la fois. "
                         "L'ensemble de prediction et l'orientation restent pertinents.",
        "result": "Decision de triage",
        "refer": "ORIENTER / ESCALADER",
        "monitor": "Surveiller - pas de declenchement d'orientation",
        "next_test": "Test suivant recommande",
        "calibrated": "Decision calibree (paludisme, dengue)",
        "context_only": "Affiche a titre indicatif (non calibre)",
        "context_note": "La fievre jaune et la typhoide ont trop peu de cas ici (12 et 29) pour "
                        "calibrer un seuil fiable ; elles n'entrent jamais dans l'ensemble de "
                        "prediction et ne declenchent pas d'orientation ; leurs probabilites sont "
                        "indicatives.",
        "pred_set": "Ensemble de prediction (ne peut etre exclu a {conf}% de confiance)",
        "in_set": "DANS L'ENSEMBLE",
        "ruled_out": "exclu",
        "kept_if": "conserve si &#8805;{thr}%",
        "needs": "exige &#8805;{thr}%",
        "empty_set": "vide",
        "why": "Pourquoi cette evaluation",
        "why_help": "Principaux signes augmentant (+) ou diminuant (-) chaque maladie calibree, "
                    "attribues a un modele par maladie sur les memes signes. Ceci explique le sens "
                    "de la preuve ; la probabilite affichee provient de l'ensemble de chaines.",
        "why_increase": "augmente",
        "why_decrease": "diminue",
        "why_none": "Aucun signe coche ne modifie sensiblement cette maladie.",
        "equity": "Note d'equite",
        "equity_text": "Un audit d'equite (Section 5.6) a montre que la regle d'orientation se "
                       "declenche a une sensibilite similaire selon le sexe et les tranches d'age "
                       "les mieux dotees, mais le groupe des moins de cinq ans est trop petit pour "
                       "etre certifie. Lisez le comportement par sous-groupe avec cette reserve.",
        "recal_title": "Generalisation : pourquoi ce modele doit etre recalibre localement",
        "recal_intro": "La garantie de couverture conforme a 90% ne tient que si les populations de "
                       "calibration et de deploiement coincident. La cohorte couvre deux districts "
                       "sanitaires aux dossiers cliniques tres differents ; le tableau ci-dessous "
                       "traite donc chaque district comme un substitut d'une nouvelle structure : "
                       "il montre la couverture obtenue par le seuil calibre sur la cohorte poolee, "
                       "ou recalibre sur chaque district, puis applique a chaque district. C'est la "
                       "forme interactive de la feuille de route de la Section 6.3.",
        "recal_dengue_h": "Couverture dengue (cible 90%)",
        "recal_malaria_h": "Couverture paludisme (cible 90%)",
        "recal_calon": "Seuil calibre sur",
        "recal_at": "Couverture a",
        "recal_pooled": "Cohorte poolee (livree)",
        "recal_local": "{site} (recalibration locale)",
        "recal_takeaway": "Le seuil livre (poole) ne couvre la dengue qu'a {cov_pooled:.0f}% dans le "
                          "district {scarce} pauvre en dengue, bien en dessous de la cible de 90%, "
                          "car sa distribution de scores differe. Une recalibration sur les donnees "
                          "locales de {scarce} retablit la couverture a {cov_local:.0f}%, au prix "
                          "d'un taux d'orientation plus eleve. Le paludisme, maladie commune, se "
                          "transfere bien et reste pres de 90%. Voila pourquoi un modele deploye "
                          "voyage par recalibration locale et non par des seuils fixes ; notez que "
                          "la dengue locale de {scarce} ne repose que sur {n_scarce} cas, donc des "
                          "donnees locales granulaires sont elles-memes une condition prealable.",
        "recal_note": "Les cellules ou un seuil est evalue sur le district qui a servi a le calibrer "
                      "sont la cible intra-population et sont legerement optimistes ; les cellules "
                      "poolees et inter-districts sont les vrais chiffres de transfert. La couverture "
                      "est affichee avec le nombre de positifs d'evaluation, car les cellules dengue "
                      "reposent sur peu de cas.",
        "how_title": "Comment cela fonctionne",
        "card_title": "Fiche du modele et provenance",
        "no_signs": "Aucun signe coche. Cochez les signes presentes pour afficher le triage.",
        "retrain_note": "Modele fige introuvable ; reconstruit a partir de la cohorte. Lancez "
                        "export_model.py pour le mettre en cache.",
    },
}

# French labels for the twelve questionnaire signs (English falls back to the override map).
FR_LABEL = {
    "oliguria": "Oligurie (urines reduites)",
    "shiver_or_cold_sensation": "Frissons",
    "back_pain_rachiodynia": "Douleur dorsale (rachialgie)",
    "positive_tourniquet_test": "Test du lacet positif",
    "frequent_urination": "Pollakiurie (mictions frequentes)",
    "joint_pain": "Douleurs articulaires",
    "osteoarthritis": "Douleur osteo-articulaire",
    "mucosal_skin_pallor_or_anemia": "Paleur cutaneo-muqueuse",
    "headache_2": "Cephalees",
    "chaist_pain": "Douleur thoracique",
    "constipation": "Constipation",
    "multiple_convulsions": "Convulsions",
}

# Illustrative (hypothetical, not real patient) presets, by feature-column name. Chosen so the
# two presentations demonstrate the contrasting triage paths cleanly: the malaria-like set
# resolves to a confident {Malaria} with no referral, while the dengue-suspicious set has malaria
# ruled out yet keeps dengue in the conformal set and therefore refers (the Section 5.4 behaviour).
PRESETS = {
    "preset_malaria": ["oliguria", "back_pain_rachiodynia", "frequent_urination", "headache_2",
                       "shiver_or_cold_sensation", "joint_pain", "mucosal_skin_pallor_or_anemia",
                       "chaist_pain"],
    "preset_dengue": ["frequent_urination", "headache_2", "shiver_or_cold_sensation",
                      "positive_tourniquet_test", "joint_pain", "chaist_pain",
                      "multiple_convulsions"],
}


@st.cache_resource(show_spinner="Loading the SENTINEL model ...")
def load_engine():
    """Load the frozen model.joblib; if absent, rebuild from the cohort (cached for the session)."""
    if os.path.exists(MODEL_PATH):
        import joblib
        return joblib.load(MODEL_PATH), False
    eng = core.build_engine(_resolve_data_path())
    payload = dict(
        ecc=eng["ecc"], br_models=eng["br_models"], qhat=eng["qhat"],
        next_test=eng["next_test"], deploy_cols=eng["deploy_cols"],
        question_cols=eng["question_cols"], feature_groups=eng["feature_groups"],
        alpha=eng["alpha"], labels=eng["labels"], disp=eng["disp"],
        recal_data=eng["recal_data"],
        conformal_summary={l: dict(qhat=eng["conf"][l]["qhat"],
                                   coverage=eng["conf"][l]["coverage"],
                                   n_test_pos=eng["conf"][l]["n_test_pos"],
                                   set_rate=eng["conf"][l]["set_rate"]) for l in core.CONF_LABELS},
    )
    return payload, True


def _resolve_data_path() -> str:
    for cand in (os.path.join(HERE, "..", "data.csv"), os.path.join(HERE, "data.csv"), "data.csv"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("data.csv not found for the in-session rebuild.")


@st.cache_data
def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def sign_label(col: str, lang: str) -> str:
    if lang == "fr" and col in FR_LABEL:
        return FR_LABEL[col]
    return core.label_for(col)


def prob_bar(label, prob, status, p_min, disp, t):
    """One disease row: probability bar, percentage, and calibrated/descriptive status."""
    pct = int(round(prob * 100))
    if status == "uncalibrated":
        right = '<span style="color:#999">-</span>'
        track = "#f3f3f3"
    else:
        thr = int(round(p_min * 100))
        if status == "in set":
            right = (f'<b style="color:#b23a48">{t["in_set"]}</b> '
                     f'<span style="color:#888">({t["kept_if"].format(thr=thr)})</span>')
        else:
            right = (f'<span style="color:#3a7d44">{t["ruled_out"]}</span> '
                     f'<span style="color:#888">({t["needs"].format(thr=thr)})</span>')
        track = "#eee"
    return (
        f'<div style="margin:5px 0;display:flex;align-items:center;font-size:14px">'
        f'<div style="width:120px;text-align:right;padding-right:10px;font-weight:600">{disp}</div>'
        f'<div style="flex:1;background:{track};border-radius:3px;height:20px;position:relative">'
        f'<div style="width:{max(pct,1)}%;background:{BAR_COLOR[label]};height:20px;'
        f'border-radius:3px"></div></div>'
        f'<div style="width:230px;padding-left:10px">{pct}% &nbsp; {right}</div></div>')


def main():
    st.set_page_config(page_title="SENTINEL triage prototype", page_icon="🩺", layout="wide")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2rem; max-width: 1150px;}
          .stCheckbox {margin-bottom: -0.5rem;}
          div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    payload, retrained = load_engine()
    meta = load_meta()
    LABELS = payload["labels"]
    DISP = payload["disp"]
    qhat = payload["qhat"]
    deploy_cols = payload["deploy_cols"]
    question_cols = payload["question_cols"]
    feature_groups = payload["feature_groups"]
    next_test = payload["next_test"]
    conf_pct = int((1 - payload["alpha"]) * 100)

    # ---- sidebar: settings -----------------------------------------------------------
    with st.sidebar:
        lang = "en"
        t = TEXT["en"]
        st.markdown("### SENTINEL")
        lang_choice = st.radio(TEXT["en"]["language"] + " / Langue", ["English", "Francais"],
                               horizontal=True, label_visibility="collapsed")
        lang = "fr" if lang_choice == "Francais" else "en"
        t = TEXT[lang]

        st.markdown(f"#### {t['sidebar_settings']}")
        mode = st.radio(t["form_mode"], [t["short_form"], t["full_form"]],
                        help=t["short_help"])
        short = (mode == t["short_form"])

        st.markdown(f"#### {t['demographics']}")
        age = st.number_input(t["age"], min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        sex = st.radio(t["sex"], [t["sex_unknown"], t["sex_female"], t["sex_male"]],
                       horizontal=False)

        st.markdown(f"#### {t['presets']}")
        st.caption(t["preset_help"])
        preset = st.selectbox(
            t["presets"],
            [t["preset_none"], t["preset_malaria"], t["preset_dengue"]],
            label_visibility="collapsed",
        )

    # Apply a preset only when the selection CHANGES, so it seeds the checkboxes once and the
    # user can then freely toggle them (re-applying every rerun would lock the boxes).
    if preset != st.session_state.get("_last_preset"):
        st.session_state["_last_preset"] = preset
        chosen_preset = set()
        if preset == t["preset_malaria"]:
            chosen_preset = set(PRESETS["preset_malaria"])
        elif preset == t["preset_dengue"]:
            chosen_preset = set(PRESETS["preset_dengue"])
        all_sign_cols = set(question_cols) | set(feature_groups["symptom"] + feature_groups["history"])
        for c in all_sign_cols:
            st.session_state[f"sig_{c}"] = (c in chosen_preset)

    # ---- header ----------------------------------------------------------------------
    st.title("🩺 SENTINEL")
    st.caption(t["tagline"])
    st.error(t["disclaimer"], icon="⚠️")
    if retrained:
        st.info(t["retrain_note"], icon="ℹ️")

    left, right = st.columns([1, 1.15], gap="large")

    # ---- left: clinical sign input ---------------------------------------------------
    with left:
        st.subheader(t["signs_present"])
        st.caption(t["signs_caption"])
        present = []
        if short:
            cols = st.columns(2)
            for i, c in enumerate(question_cols):
                with cols[i % 2]:
                    if st.checkbox(sign_label(c, lang), key=f"sig_{c}"):
                        present.append(c)
        else:
            sympt = [c for c in (feature_groups["symptom"] + feature_groups["history"])
                     if c in deploy_cols]
            with st.expander(t["signs_present"], expanded=True):
                cols = st.columns(2)
                for i, c in enumerate(sorted(sympt, key=lambda x: sign_label(x, lang))):
                    with cols[i % 2]:
                        if st.checkbox(sign_label(c, lang), key=f"sig_{c}"):
                            present.append(c)

    # ---- build the patient record and score ------------------------------------------
    record = {c: 1.0 for c in present}
    # symptom_burden is a function of the marked symptoms (Section 3.3); compute it rather than
    # leaving it absent, so the derived feature is internally consistent with the ticks.
    sym_marked = [c for c in present if c in feature_groups["symptom"]]
    if "symptom_burden" in deploy_cols:
        record["symptom_burden"] = float(len(sym_marked))
    if age and age > 0:
        if "age_years" in deploy_cols:
            record["age_years"] = float(age)
        if "age_band" in deploy_cols:
            record["age_band"] = float(0 if age < 5 else (1 if age < 15 else 2))
    if sex != t["sex_unknown"] and "is_female" in deploy_cols:
        record["is_female"] = 1.0 if sex == t["sex_female"] else 0.0

    probs = core.score_record(record, payload["ecc"], deploy_cols)
    r = core.triage(probs, qhat, next_test)

    # ---- right: triage decision card -------------------------------------------------
    with right:
        st.subheader(t["result"])
        if not present:
            st.warning(t["no_signs"], icon="👈")

        if r["refer"]:
            st.markdown(
                f'<div style="background:#b23a48;color:#fff;padding:12px 16px;border-radius:8px;'
                f'font-size:18px;font-weight:700">&#9888; {t["refer"]}</div>',
                unsafe_allow_html=True)
            st.markdown(f"**{t['next_test']}:** `{next_test}`")
        else:
            st.markdown(
                f'<div style="background:#3a7d44;color:#fff;padding:12px 16px;border-radius:8px;'
                f'font-size:18px;font-weight:700">&#10003; {t["monitor"]}</div>',
                unsafe_allow_html=True)

        if 0 < len(present) < 6:
            st.caption(t["low_prob_note"])

        st.markdown(f'<div style="font-size:12px;color:#777;text-transform:uppercase;'
                    f'letter-spacing:.04em;margin:14px 0 2px">{t["calibrated"]}</div>',
                    unsafe_allow_html=True)
        cal = [l for l in LABELS if r["status"][l] != "uncalibrated"]
        desc = [l for l in LABELS if r["status"][l] == "uncalibrated"]
        st.markdown("".join(prob_bar(l, r["prob"][l], r["status"][l], r["p_min"][l], DISP[l], t)
                            for l in cal), unsafe_allow_html=True)

        setstr = ", ".join(DISP[l] for l in r["pred_set"]) or t["empty_set"]
        st.markdown(f'<div style="font-size:13px;color:#333;margin-top:8px">'
                    f'{t["pred_set"].format(conf=conf_pct)}: <b>{{{setstr}}}</b></div>',
                    unsafe_allow_html=True)

        st.markdown('<hr style="border:none;border-top:1px dashed #ddd;margin:12px 0 6px">',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#777;text-transform:uppercase;'
                    f'letter-spacing:.04em;margin:0 0 2px">{t["context_only"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(prob_bar(l, r["prob"][l], r["status"][l], r["p_min"][l], DISP[l], t)
                            for l in desc), unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:#999;margin-top:4px">{t["context_note"]}</div>',
                    unsafe_allow_html=True)

    # ---- why panel (SHAP) ------------------------------------------------------------
    if present:
        with st.expander(f"🔎 {t['why']}", expanded=False):
            st.caption(t["why_help"])
            wc = st.columns(len(cal))
            for k, l in enumerate(cal):
                with wc[k]:
                    st.markdown(f"**{DISP[l]}**")
                    try:
                        tops = core.shap_top_symptoms(l, record, payload["br_models"],
                                                      deploy_cols, feature_groups, top_k=6)
                    except Exception:
                        tops = []
                    tops = [(name, v) for name, v in tops]
                    if not tops:
                        st.caption(t["why_none"])
                    for name, v in tops:
                        arrow = "▲" if v > 0 else "▼"
                        color = "#b23a48" if v > 0 else "#3a7d44"
                        word = t["why_increase"] if v > 0 else t["why_decrease"]
                        st.markdown(
                            f'<div style="font-size:13px;margin:2px 0">'
                            f'<span style="color:{color}">{arrow}</span> {name} '
                            f'<span style="color:#999;font-size:11px">({word})</span></div>',
                            unsafe_allow_html=True)

    st.info(f"**{t['equity']}.** {t['equity_text']}", icon="⚖️")

    # ---- generalisation: local recalibration demo (Section 6.2) -----------------------
    recal_data = payload.get("recal_data")
    if recal_data is not None:
        with st.expander(f"🌍 {t['recal_title']}"):
            st.caption(t["recal_intro"])
            rm = core.recalibration_matrix(recal_data)
            sites = rm["sites"]

            def coverage_table(label):
                rows = {}
                for cal in rm["cal_sources"]:
                    src = (t["recal_pooled"] if cal == "pooled"
                           else t["recal_local"].format(site=cal))
                    rows[src] = {}
                    for s in sites:
                        cell = rm["matrix"][label][cal][s]
                        col = f"{t['recal_at']} {s}"
                        if cell is None:
                            rows[src][col] = "n/a"
                        else:
                            rows[src][col] = f"{cell['coverage']*100:.0f}% (n={cell['n_eval_pos']})"
                return pd.DataFrame(rows).T

            st.markdown(f"**{t['recal_dengue_h']}**")
            st.dataframe(coverage_table("dengue"), width="stretch")
            st.markdown(f"**{t['recal_malaria_h']}**")
            st.dataframe(coverage_table("malaria"), width="stretch")

            scarce = min(sites, key=lambda s: rm["n_positives"]["dengue"][s])
            den = rm["matrix"]["dengue"]
            cov_pooled = den["pooled"][scarce]["coverage"] * 100
            cov_local = den[scarce][scarce]["coverage"] * 100
            n_scarce = rm["n_positives"]["dengue"][scarce]
            st.markdown(t["recal_takeaway"].format(cov_pooled=cov_pooled, cov_local=cov_local,
                                                   scarce=scarce, n_scarce=n_scarce))
            st.caption(t["recal_note"])

    # ---- how it works ----------------------------------------------------------------
    with st.expander(f"ℹ️ {t['how_title']}"):
        if lang == "fr":
            st.markdown(
                "1. **Profil multi-maladies** : un ensemble de chaines de classifieurs predit "
                "conjointement le paludisme, la dengue, la typhoide et la fievre jaune a partir "
                "des seuls signes cliniques (Section 4).\n"
                "2. **Ensemble de prediction calibre** : la prediction conforme par maladie "
                "conserve une maladie tant qu'elle ne peut etre exclue a 90% de confiance ; "
                "garantie uniquement pour le paludisme et la dengue (Section 5.3).\n"
                "3. **Regle d'orientation** : si la dengue ne peut etre exclue, orienter pour "
                "confirmation. Mieux vaut un test de trop qu'une dengue manquee (Section 5.4).\n"
                "4. **Test suivant** : le classement valeur-de-l'information identifie le test qui "
                "reduit le plus l'incertitude residuelle (Section 5.5).")
        else:
            st.markdown(
                "1. **Multi-disease profile** - an ensemble of classifier chains jointly predicts "
                "malaria, dengue, typhoid and yellow fever from clinical signs alone (Section 4).\n"
                "2. **Calibrated prediction set** - per-label conformal prediction keeps a disease "
                "in the set while it cannot be ruled out at 90% confidence; guaranteed only for "
                "malaria and dengue (Section 5.3).\n"
                "3. **Referral rule** - if dengue cannot be ruled out, refer for confirmation. An "
                "unnecessary test is preferable to a missed dengue case (Section 5.4).\n"
                "4. **Next test** - the value-of-information ranking identifies the single test that "
                "most reduces the residual uncertainty (Section 5.5).")

    # ---- model card ------------------------------------------------------------------
    with st.expander(f"📋 {t['card_title']}"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Cohort", f"N = {meta.get('n_patients', 299)}", "Burkina Faso, single region")
        c2.metric("Non-malarial dengue recall",
                  f"{meta.get('recall_nmd_ecc', 0.357)*100:.0f}% / "
                  f"{meta.get('recall_nmd_br', 0.429)*100:.0f}%", "ECC / BR (lit. ~0-5.5%)")
        c3.metric("Conformal coverage",
                  f"{meta.get('coverage_malaria', 0.92)*100:.0f}% / "
                  f"{meta.get('coverage_dengue', 0.89)*100:.0f}%", "Malaria / Dengue (target 90%)")
        st.markdown(
            f"- **Referral path** catches {meta.get('referral_caught', 24)} of "
            f"{meta.get('referral_dengue_n', 27)} dengue on the held-out split, versus "
            f"{meta.get('naive_caught', 8)} for a naive 0.5 detector.\n"
            f"- **Recommended next test:** {meta.get('next_test', next_test)} "
            f"(value-of-information, population level).\n"
            f"- **Inputs:** clinical signs (and optional age and sex); the model is symptoms-only, "
            f"so it runs with every laboratory field blank.\n"
            f"- **Limitations:** single-region cohort, paediatric-skewed, severely imbalanced "
            f"(yellow fever n=12). Conformal guarantees hold under exchangeability and would need "
            f"local recalibration before use elsewhere. Proof of concept, not a clinical device.")
        if meta.get("built_at"):
            st.caption(f"Model frozen {meta['built_at']} (seed {meta.get('seed', core.SEED)}). "
                       f"Faithfulness to the research notebook is asserted at export time.")


if __name__ == "__main__":
    main()
