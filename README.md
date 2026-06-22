# Multi-Disease Test Evaluation: Cardiovascular-Kidney-Metabolic (CKM) Risk
## From Single-Disease Decision Curve Analysis to a Correlated Multi-Organ Problem

**Author:** SeyedehPari Yadegaritotshami
**Background:** MSc Health Economics, Management & Policy — Università di Bologna
**Supervisor:** Prof. Pietro Biroli
**Contact:** yadegari.parii@gmail.com · [LinkedIn] · [GitHub]

A close family experience with delayed diagnosis is part of what drew me to this question — but the project below is built to stand on its own methodological merits.

---

## Research Question

Health technology assessment has a mature toolkit for evaluating a diagnostic test that targets **one** disease: Decision Curve Analysis (DCA), sensitivity and specificity, likelihood ratio decomposition. This toolkit assumes one disease, one decision, one set of harms and benefits.

That assumption breaks down for modern multi-disease tests — liquid biopsies that screen for 50 cancer types from one blood draw, AI radiology systems that flag 100+ abnormalities from one scan. The technical capability to detect multiple conditions from a single sample already exists. What does not yet exist is a correct method for evaluating whether using such a test produces real clinical benefit, once false-positive harms compound across correlated conditions.

**This project asks:** does applying standard single-disease DCA independently to each target of a multi-disease test, then summing the results, give the correct estimate of clinical value — or does it systematically overestimate it?

This is direct preparation for the [MERIT PhD position](https://utwentecareers.nl/en/vacancies/2456/) at the University of Twente — *Multi-disease tEsting: fRom Information to impacT* — supervised by Prof. Dr. Erik Koffijberg and Prof. Dr. Mariska Leeflang.

---

## Why Cardiovascular-Kidney-Metabolic (CKM) Syndrome

Three conditions were selected — diabetes, cardiovascular risk, and kidney disease — not because they are statistically convenient, but because they form a recognized clinical syndrome with shared biological mechanisms. Insulin resistance damages vascular tissue; vascular damage and abnormal lipids impair kidney filtration over time; reduced kidney function in turn worsens cardiovascular risk through fluid retention and altered mineral metabolism. One underlying metabolic disturbance produces correlated signals across three organ systems — structurally the same pattern by which a single biological process (circulating tumor DNA) produces correlated signals across multiple cancer types in a liquid biopsy.

A recent NHANES-based study has already paired machine learning with Decision Curve Analysis specifically to evaluate CKM biomarkers, confirming this as an active, clinically recognized methodological direction rather than a constructed analogy.

---

## Dataset

**CDC NHANES 2017–2018** (National Health and Nutrition Examination Survey) — publicly available, no login required. Source: [wwwn.cdc.gov/nchs/nhanes/2017-2018](https://wwwn.cdc.gov/nchs/nhanes/2017-2018/)

NHANES was selected over commonly used teaching datasets because it provides real, directly-measured biomarkers — not self-reported symptoms — from a nationally representative sample of US adults. This produces realistic model performance rather than the artificially inflated results common in toy datasets.

### Files merged

| File | Variable(s) | Role |
|---|---|---|
| `DEMO_J.xpt` | Age (RIDAGEYR) | Covariate |
| `DIQ_J.xpt` | Diabetes diagnosis (DIQ010) | Outcome 1 — Metabolic |
| `BPX_J.xpt` | Systolic BP (BPXSY1) | Outcome 2 — Cardiovascular (component) |
| `TCHOL_J.xpt` | Total cholesterol (LBXTC) | Outcome 2 — Cardiovascular (component) |
| `HDL_J.xpt` | HDL cholesterol (LBDHDD) | Predictor |
| `TRIGLY_J.xpt` | Triglycerides (LBXTR), LDL (LBDLDL) | Predictor (fasting subsample) |
| `BMX_J.xpt` | BMI (BMXBMI) | Predictor |
| `BIOPRO_J.xpt` | Creatinine (LBXSCR) | Outcome 3 — Kidney (used to calculate eGFR) |

Every predictor is either directly measured from blood or taken at the same clinical visit (BMI). Age is the sole demographic covariate, consistent with every published CKM and cardiovascular risk model reviewed during this project's design. No predictor is also used to define an outcome — total cholesterol and creatinine define outcomes and are therefore excluded from the predictor set, preventing outcome leakage.

### Outcome definitions

| Outcome | Definition | Clinical threshold |
|---|---|---|
| Diabetes | Doctor-diagnosed diabetes | DIQ010 = 1 |
| Cardiovascular risk | Elevated BP or high total cholesterol | Systolic BP ≥ 130 mmHg OR total cholesterol ≥ 200 mg/dL |
| Kidney disease | Reduced filtration function | eGFR < 60 mL/min/1.73m² (CKD-EPI 2021 equation) |

---

## Methods

**Stage 1 — Data preparation.** Merge all 8 NHANES files on SEQN. Filter to adults (age ≥ 18). Classify every variable's missingness mechanism — near-universal, fasting-subsample, or survey skip-pattern — before deciding which variables are required in the main analytic sample. Skip-pattern variables (e.g. occupation-specific physical activity) are excluded from required fields; including them was tested and found to reduce the analytic sample by over 90%.

**Stage 2 — Outcome construction.** Build the three CKM outcomes from standard clinical thresholds. Calculate eGFR from serum creatinine using the CKD-EPI 2021 equation.

**Stage 3 — Correlation analysis.** Test whether the three outcomes are positively correlated in the NHANES sample — the empirical premise for the rest of the analysis.

**Stage 4 — Single-disease DCA (baseline).** Implement standard DCA independently for each outcome, using a logistic regression risk score built from the shared blood-based predictor panel. This represents current HTA practice.

**Stage 5 — The multi-disease problem.** Demonstrate that naively summing single-disease net benefit across all three outcomes overestimates the test's value, because patients flagged for multiple correlated conditions are counted as multiple false positives when they represent one patient, one clinical encounter, one harm.

**Stage 6 — Patient-level correction.** Evaluate net benefit at the patient level (flagged for any condition vs. flagged for none) instead of summing independently across diseases, and quantify the gap between the two approaches.

---

## Results

| Metric | Value |
|---|---|
| Final analytic sample (N) | 4,703 |
| Diabetes prevalence | 14.8% |
| Cardiovascular risk prevalence | 57.6% |
| Kidney disease prevalence (eGFR < 60) | 8.1% |
| Patients carrying ≥ 2 of the three conditions | 14.3% (673 / 4,703) |
| Pairwise correlation — diabetes × kidney | phi = 0.17, OR = 3.68 (p < 0.001) |
| Pairwise correlation — cardiovascular × kidney | phi = 0.08, OR = 1.81 (p < 0.001) |
| Pairwise correlation — diabetes × cardiovascular | phi = 0.05, OR = 1.34 (p < 0.001) |
| Naive summation vs. patient-level net benefit | diverge; magnitude of the gap depends on the decision threshold |

All three CKM outcomes are positively and statistically significantly correlated, with the strongest association between diabetes and kidney disease — consistent with diabetic kidney disease being the leading cause of reduced kidney function. Because the conditions cluster together (14.3% of patients carry two or more), evaluating each disease independently and summing the results counts these patients more than once.

Single-disease decision curve analysis shows that each of the three risk models adds net benefit over both treat-all and treat-none across clinically relevant thresholds. When the three single-disease net benefits are naively summed and compared with a patient-level evaluation (flagged for any condition vs. flagged for none), the two approaches diverge, and the magnitude of the difference depends on the chosen threshold. The point is not a single overestimation figure but the more basic one: summing single-disease net benefit is not equivalent to evaluating the test at the patient level. Establishing the correct way to evaluate net benefit for a multi-disease test is exactly the methodological problem the MERIT project is designed to solve.


---

## What This Project Does Not Solve

The patient-level correction handles the simplest case — treating all three conditions as equally important at one shared threshold. Four problems remain open, and these are the research questions MERIT is funded to answer:

1. **Disease-specific harm weights.** A false positive for kidney disease likely triggers a different, more invasive follow-up pathway than a false positive for elevated cholesterol. This project treats all false positives as equally harmful.
2. **Disease combination harm.** Being flagged for two or three CKM components simultaneously may cause compounding clinical and psychological harm beyond the sum of each condition's individual harm.
3. **Joint probability modelling.** This project uses independent per-outcome predicted probabilities rather than modelling the full joint distribution across all three correlated outcomes.
4. **Varying thresholds per condition.** Each CKM component may warrant a different clinically appropriate decision threshold; this project uses one shared threshold across all three.

---

## Technical Skills Demonstrated

| Tool | Usage |
|---|---|
| Python | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` |
| Clinical calculation | CKD-EPI eGFR equation implemented from the published formula |
| Survey data handling | Multi-file merge on shared identifier; structural vs. random missingness documentation |
| DCA implementation | Net benefit formula implemented directly, not from a pre-built library |
| Cross-validation | Stratified k-fold with out-of-fold predictions to prevent data leakage |
| Stata + Mplus | Used in thesis — BRR weighting, WLSMV, plausible-value pooling via Rubin's rules |

---

## Connection to My Master's Thesis

My MSc thesis used Structural Equation Modelling with a latent health literacy factor, WLSMV estimation, and plausible-value pooling across 10 datasets via Rubin's rules to decompose how socioeconomic status affects self-rated health through two distinct pathways — health literacy and income. The methodological connection here is direct: both projects require decomposing a multi-pathway outcome into attributable components while correctly handling correlated mediators or outcomes, rather than assuming independence.

---

## Key References

- Vickers AJ, Elkin EB (2006). Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565–574.
- Deeks JJ, Bossuyt PM, Leeflang MM, Takwoingi Y (2023). *Cochrane Handbook for Systematic Reviews of Diagnostic Test Accuracy*. Wiley.
- Koffijberg H et al. (2024). CHEERS-VOI reporting guideline. *Medical Decision Making*, 44(2).
- CKD-EPI 2021 Creatinine Equation for estimated glomerular filtration rate.

---

## About the MERIT Project

MERIT will build the framework this project demonstrates is needed: a complete methodology for health and economic evaluation of multi-disease tests, tested against two real case studies — a 50-cancer liquid biopsy and a 100-abnormality AI radiology system.

This project establishes the evaluation problem empirically, using a real, clinically coherent multi-organ syndrome. It does not solve it. That is the work I want to do next.
