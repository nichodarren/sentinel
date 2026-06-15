# SENTINEL triage app - cognitive walkthrough (usability evaluation)

This is a structured **cognitive walkthrough** of the deployed SENTINEL triage interface, the
lightweight usability-evaluation method appropriate to a proof-of-concept artefact. It is an
**expert inspection performed by the project team**, not a field study with naive users; that
distinction is stated up front because it bounds the strength of the evidence. A cognitive
walkthrough steps an evaluator through each action a target user would take to complete a task,
asking, at every step, the four standard questions of Wharton et al.:

1. Will the user try to achieve the right effect? (Is the goal clear?)
2. Will the user notice that the correct action is available? (Is the control visible?)
3. Will the user associate the correct action with the effect they want?
4. After the action, will the user see that progress has been made? (Is the feedback clear?)

## Target user and context

A frontline community health worker (CHW) or clinician in a low-resource clinic during a
post-disaster surge of undifferentiated febrile illness. Time per patient is short, point-of-care
confirmatory tests are scarce or unavailable, and the user is clinically trained but not a
statistician. The relevant goal is triage: decide who to refer and which confirmatory test would
help, not to reach a diagnosis.

## Task 1: assess a febrile patient and decide whether to refer

| Step | Action | Q1 goal | Q2 control visible | Q3 action-effect link | Q4 feedback | Verdict |
|---|---|---|---|---|---|---|
| 1 | Choose the short questionnaire | Clear: "Symptom entry" with the short form selected by default | The radio is the first control in the sidebar | Default already correct, no action needed | The sign list narrows to twelve | Pass |
| 2 | Tick the signs the patient presents | Clear heading "Clinical signs present" + caption "tick the signs the patient presents" | Checkboxes laid out two columns, plain clinical labels | Direct: each tick is one sign | The decision updates live (no submit needed) | Pass |
| 3 | Read the decision | Clear: a large coloured banner is the first thing in the result column | REFER / ESCALATE (red) or Monitor (green) is unmissable | The banner is the action, not a number | Banner colour and verb state the action | Pass |
| 4 | Decide which test to run | Clear when referring: "Recommended next test" sits directly under the banner | Shown only when a referral fires, so it appears exactly when relevant | Single named test, not a list | Test name rendered as a code chip | Pass |
| 5 | Understand why dengue is flagged at a low percentage | At-risk step: a non-statistical user may distrust a referral at, say, 6 percent | The "kept if >=X%" annotation and the low-probability caption are present | Requires reading the annotation to link low probability to the inclusion rule | Annotation and caption explain it, but it is secondary text | Partial: see finding F1 |

## Findings

- **F1 (addressed).** The most likely point of confusion is a referral firing while the dengue
  probability reads low (for example 6 percent). A user could read this as a contradiction. The
  design already mitigates it in three ways: the actionable banner leads and does not show a
  number, each calibrated disease shows its inclusion threshold ("kept if >=1%"), and a caption
  states that low probabilities are by design when few signs are ticked. Residual risk remains for
  a hurried user who does not read the secondary text; a future iteration could add a one-line
  plain-language reason on the banner itself ("dengue cannot be ruled out").
- **F2 (addressed).** Statistical vocabulary ("prediction set", "conformal", "coverage") could be
  opaque to a CHW. It is deliberately kept out of the primary action path: the banner and the
  recommended test carry the decision, while the set and thresholds are supporting detail. The
  separate "shown for context only" block with its explainer keeps yellow fever and typhoid from
  being mistaken for actionable.
- **F3 (works well).** The live update, the colour-coded banner, the plain-language sign labels,
  the optional French interface for the francophone setting, and the persistent non-clinical
  disclaimer all passed every step without friction. The illustrative presets let a first-time
  user see a complete decision immediately.
- **F4 (works well).** Lab-free operation is faithful to the field condition: the task completes
  with no laboratory entry, and the value-of-information recommendation is the only place a test
  is mentioned, framed as advice rather than a requirement.

## Mapping to the deployment success criteria

This walkthrough provides the usability evidence in the project's deployment evaluation plan
(functional, faithfulness, lab-free robustness, latency, interpretability, safety behaviour,
**usability**, equity surface, honesty). The interface supports the core triage task end to end,
the action is salient, and the two identified friction points are already mitigated in the design.

## Honest limitations and next step

A team cognitive walkthrough is weaker evidence than observing real users, and it shares the
evaluator's blind spots. It does not establish that an actual CHW under field conditions would
interpret the output correctly, nor that the clinical content is appropriate. Consistent with the
out-of-scope list of the evaluation plan (no real clinical or field study, no prospective
validation, no regulatory clearance), the appropriate next step is a small **proxy-CHW usability
test** with think-aloud observation and a **clinician content review**, both of which require human
participants and are therefore deferred to the work that would precede any pilot.
