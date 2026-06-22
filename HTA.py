import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import os

# Folder containing the NHANES .xpt files (and where outputs are written).
# Set this to your own path before running.
DATA_DIR = r"C:\Users\PC\Downloads"
# ********************************* Data Prepration******************************
# 1. I will import all the dataset related to outcomes and predictors
#2. Then we will merge them and doing data preparation 
#******************** PREDICTORE VARIABLES*********************************
demo = pd.read_sas(os.path.join(DATA_DIR, "DEMO_J.xpt"), format='xport', encoding='utf-8')
print(demo.shape)
demo_sel = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()
demo_sel.columns = ['SEQN', 'age', 'gender']

print(demo_sel.head())
print(demo_sel.shape)
bmx = pd.read_sas(os.path.join(DATA_DIR, "BMX_J.xpt"), format='xport', encoding='utf-8')
bmx_sel = bmx[['SEQN', 'BMXBMI']].copy()
bmx_sel.columns = ['SEQN', 'bmi']
print(bmx_sel.head())
print(bmx_sel.shape)
hdl = pd.read_sas(os.path.join(DATA_DIR, "HDL_J.xpt"), format='xport', encoding='utf-8')

hdl_sel = hdl[['SEQN', 'LBDHDD']].copy()
hdl_sel.columns = ['SEQN', 'hdl_cholesterol']

print(hdl_sel.head())
print(hdl_sel.shape)

trigly = pd.read_sas(os.path.join(DATA_DIR, "TRIGLY_J.xpt"), format='xport', encoding='utf-8')
trigly_sel = trigly[['SEQN', 'LBXTR', 'LBDLDL']].copy()
trigly_sel.columns = ['SEQN', 'triglycerides', 'ldl_cholesterol']
print(trigly_sel.head())
print(trigly_sel.shape)
#*************** Creating predictor variable with merging all of above variables***
# Merge all four predictor tables together on SEQN
predictors = demo_sel.merge(bmx_sel, on='SEQN', how='left')
predictors = predictors.merge(hdl_sel, on='SEQN', how='left')
predictors = predictors.merge(trigly_sel, on='SEQN', how='left')
print(predictors.head())
print(predictors.shape)
print(predictors.columns.tolist())
#**********************OUTCOME VARIABLES***************

diabetes= pd.read_sas(os.path.join(DATA_DIR, "DIQ_J.xpt"), format="xport", encoding="utf-8")
dia_sel = diabetes[['SEQN', 'DIQ010']].copy()
dia_sel.columns = ['SEQN', 'diabetes_raw']
print(dia_sel.head())
print(dia_sel.shape)
Kidney=pd.read_sas(os.path.join(DATA_DIR, "BIOPRO_J.xpt"),format="xport", encoding="utf-8" )
kidn_sel=Kidney[["SEQN","LBXSCR"]].copy()
kidn_sel.columns=["SEQN", "creatinine"]
print(kidn_sel.head())
print(kidn_sel.shape)
# Load blood pressure
bpx = pd.read_sas(os.path.join(DATA_DIR, "BPX_J.xpt"), format='xport', encoding='utf-8')
bpx_sel = bpx[['SEQN', 'BPXSY1']].copy()
bpx_sel.columns = ['SEQN', 'systolic_bp']
print(bpx_sel.head())
print(bpx_sel.shape)
# Load total cholesterol
tchol = pd.read_sas(os.path.join(DATA_DIR, "TCHOL_J.xpt"), format='xport', encoding='utf-8')
tchol_sel = tchol[['SEQN', 'LBXTC']].copy()
tchol_sel.columns = ['SEQN', 'total_cholesterol']
print(tchol_sel.head())
print(tchol_sel.shape)

# Merge the three outcome-related tables together
outcomes_raw = dia_sel.merge(bpx_sel, on='SEQN', how='left')
outcomes_raw = outcomes_raw.merge(tchol_sel, on='SEQN', how='left')
outcomes_raw = outcomes_raw.merge(kidn_sel, on='SEQN', how='left')
print(outcomes_raw.head())
print(outcomes_raw.shape)
print(outcomes_raw.columns.tolist())
#step 2 : merging all data and droping those who are not in both dataset
# Merge our predictor table and outcome table together into one final dataset
full_data = predictors.merge(outcomes_raw, on='SEQN', how='inner')
print(full_data.head())
print(full_data.shape)
print(full_data.columns.tolist())
# Outcome 4: Kidney disease (using a simplified eGFR proxy for now)
print(full_data['creatinine'].describe())

# CKD-EPI 2021 equation for estimated GFR
# Different formula depending on gender and whether creatinine is above or below the gender-specific cutoff (0.7 for women, 0.9 for men)

def calculate_egfr(row):
    scr = row['creatinine']
    age = row['age']
    gender = row['gender']  # 1 = Male, 2 = Female

    if pd.isna(scr) or pd.isna(age) or pd.isna(gender):
        return np.nan

    if gender == 2:  # Female
        if scr <= 0.7:
            egfr = 142 * (scr/0.7)**(-0.241) * (0.9938**age) * 1.012
        else:
            egfr = 142 * (scr/0.7)**(-1.200) * (0.9938**age) * 1.012
    else:  # Male
        if scr <= 0.9:
            egfr = 142 * (scr/0.9)**(-0.302) * (0.9938**age)
        else:
            egfr = 142 * (scr/0.9)**(-1.200) * (0.9938**age)

    return egfr

full_data['egfr'] = full_data.apply(calculate_egfr, axis=1)

print(full_data[['SEQN', 'age', 'gender', 'creatinine', 'egfr']].head(10))
print(full_data['egfr'].describe())
# Outcome 1: Diabetes
full_data['diabetes'] = np.where(full_data['diabetes_raw'] == 1, 1, np.where(full_data['diabetes_raw'].isin([2, 3]), 0, np.nan))
# Outcome 2: Hypertension
full_data['hypertension'] = np.where(full_data['systolic_bp'] >= 130, 1, np.where(full_data['systolic_bp'].notna(), 0, np.nan))
# Outcome 3: High cholesterol
full_data['high_cholesterol'] = np.where(full_data['total_cholesterol'] >= 200, 1,np.where(full_data['total_cholesterol'].notna(), 0, np.nan))
# Outcome 4: Kidney disease (from eGFR we just calculated)
full_data['kidney_disease'] = np.where(full_data['egfr'] < 60, 1, np.where(full_data['egfr'].notna(), 0, np.nan))
# Cardiovascular = elevated BP OR high cholesterol (one organ system)
bp_high   = full_data['systolic_bp'] >= 130
chol_high = full_data['total_cholesterol'] >= 200
positive  = bp_high | chol_high
negative  = (full_data['systolic_bp'] < 130) & (full_data['total_cholesterol'] < 200)
full_data['cardiovascular'] = np.where(positive, 1, np.where(negative, 0, np.nan))
# Check all three outcomes together
for outcome in ['diabetes', 'cardiovascular', 'kidney_disease']:
    n_yes = (full_data[outcome] == 1).sum()
    n_no = (full_data[outcome] == 0).sum()
    n_miss = full_data[outcome].isna().sum()
    print(f"{outcome:<20} {n_yes:.0f} positive | {n_no:.0f} negative | {n_miss:.0f} missing")
    #filtering to adults
    print(f"Before age filter: {len(full_data):,} people")

full_data = full_data[full_data['age'] >= 18].copy()

print(f"After age filter (adults only): {len(full_data):,} people")

# Re-check our three outcomes now that children are removed
for outcome in ['diabetes', 'cardiovascular', 'kidney_disease']:
    n_yes = (full_data[outcome] == 1).sum()
    n_no = (full_data[outcome] == 0).sum()
    n_miss = full_data[outcome].isna().sum()
    print(f"{outcome:<20} {n_yes:.0f} positive | {n_no:.0f} negative | {n_miss:.0f} missing")
    # Create a flag: did this person have a missing kidney_disease value?
full_data['kidney_missing'] = full_data['kidney_disease'].isna().astype(int)

# Compare average age between people with missing vs non-missing kidney data
print("Average age - kidney data missing vs present:")
print(full_data.groupby('kidney_missing')['age'].mean())

print("\nGender breakdown - kidney data missing vs present:")
print(full_data.groupby('kidney_missing')['gender'].value_counts(normalize=True))

print("\nBMI - kidney data missing vs present:")
print(full_data.groupby('kidney_missing')['bmi'].mean())
full_data['htn_missing'] = full_data['hypertension'].isna().astype(int)
print("Average age - hypertension missing vs present:")
print(full_data.groupby('htn_missing')['age'].mean())
print("\nGender breakdown - hypertension missing vs present:")
print(full_data.groupby('htn_missing')['gender'].value_counts(normalize=True))
predictor_cols = ['age', 'gender', 'bmi', 'hdl_cholesterol', 'triglycerides', 'ldl_cholesterol']

for col in predictor_cols:
    n_missing = full_data[col].isna().sum()
    print(f"{col:<20} missing for {n_missing:,} people")

# Final required variables for the MAIN analysis
# (triglycerides and LDL excluded - too much structural missingness,
#  reserved separately for an optional fasting-subsample sensitivity check)

required_predictors = ['age', 'gender', 'bmi', 'hdl_cholesterol']
required_outcomes = ['diabetes', 'cardiovascular', 'kidney_disease']

all_required = required_predictors + required_outcomes

print("Missingness on each required variable:")
for col in all_required:
    n_missing = full_data[col].isna().sum()
    print(f"  {col:<20} missing for {n_missing:,} people")

# Building the main clean dataset
before = len(full_data)
main_data = full_data.dropna(subset=all_required).copy()
after = len(main_data)
print(f"\nBefore dropping: {before:,} adults")
print(f"After dropping any row missing a required variable: {after:,}")
print(f"Retention rate: {after/before*100:.1f}%")
print(f"Final analytic sample: N = {len(main_data):,}\n")

print("Disease prevalences in final sample:")
for outcome in required_outcomes:
    prev = main_data[outcome].mean() * 100
    print(f"  {outcome:<20} {prev:.1f}%")

print("\nPredictor summary statistics:")
print(main_data[required_predictors].describe().round(2))

main_data.to_csv(os.path.join(DATA_DIR, "ckm_main_clean.csv"), index=False)
print(os.path.join(DATA_DIR, "ckm_main_clean.csv"))

##********************Data analysis***********************
df=pd.read_csv(os.path.join(DATA_DIR, "ckm_main_clean.csv"))
print(f"Loaded analytic sample: N={len(df):,}")
outcomes=["diabetes", "cardiovascular", "kidney_disease"]
print("Prevalence of each CKM outcome")
for o in outcomes:
    print(f"  {o:<16}{df[o].mean()*100:5.1f}%")
    df["n_conditions"] = df[outcomes].sum(axis=1)
print(df["n_conditions"].value_counts().sort_index())
#*****scipy's contingency-table test******
from scipy.stats import chi2_contingency
def tetrachoric_approx(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return np.cos(np.pi / (1 + np.sqrt((a * d) / (b * c))))
pairs = [('diabetes', 'cardiovascular'),
         ('diabetes', 'kidney_disease'),
         ('cardiovascular', 'kidney_disease')]

for x, y in pairs:
    t = pd.crosstab(df[x], df[y])
    a = t.loc[1, 1]; b = t.loc[1, 0]; c = t.loc[0, 1]; d = t.loc[0, 0]
    phi  = df[[x, y]].corr().iloc[0, 1]
    odds = (a * d) / (b * c) if b * c else np.inf
    tet  = tetrachoric_approx(a, b, c, d)
    chi2, p, _, _ = chi2_contingency(t)
    print(f"{x} x {y}:  phi={phi:+.3f}  OR={odds:.2f}  tetrachoric~{tet:+.3f}  p={p:.2e}")
phi_matrix = df[outcomes].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(phi_matrix, annot=True, fmt=".3f", vmin=0, vmax=0.4,
            cmap="Reds", square=True)
plt.title("CKM outcome correlation (phi)")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "stage3_correlation_heatmap.png"), dpi=150)
plt.show()

outcomes = ['diabetes', 'cardiovascular', 'kidney_disease']
N = len(df)
print(f"Analytic sample: N = {N:,}\n")

# ===============================================================
# STAGE 4a - Out-of-fold predicted risk for each outcome
# ===============================================================
# Predictors: shared blood/demographic panel. Total cholesterol and
# creatinine are deliberately excluded - they define outcomes.
predictors = ['age', 'gender', 'bmi', 'hdl_cholesterol']
X = df[predictors]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

print("Building out-of-fold risk scores")
for o in outcomes:
    y = df[o]
    df['risk_' + o] = cross_val_predict(model, X, y, cv=cv,
                                        method='predict_proba')[:, 1]
    print(f"  risk_{o:<16} mean={df['risk_'+o].mean():.3f}  "
          f"(prevalence={y.mean():.3f})")
print()

# ===============================================================
# Net benefit building blocks
# ===============================================================
def net_benefit_model(y_true, risk, pt):
    """NB of acting on a risk score at threshold pt:
       flag if risk >= pt, then TP/N - (FP/N) * pt/(1-pt)."""
    flag = risk >= pt
    tp = np.sum(flag & (y_true == 1))
    fp = np.sum(flag & (y_true == 0))
    return tp / N - (fp / N) * (pt / (1 - pt))

def net_benefit_all(y_true, pt):
    """NB of treating everyone (flag = True for all)."""
    prev = np.mean(y_true == 1)
    return prev - (1 - prev) * (pt / (1 - pt))

# Threshold grid (avoid 0 and 1 where the odds term misbehaves)
thresholds = np.arange(0.01, 0.51, 0.01)

# ===============================================================
# STAGE 4b - Single-disease decision curves
# ===============================================================
nb_model = {o: np.array([net_benefit_model(df[o], df['risk_'+o], pt)
                         for pt in thresholds]) for o in outcomes}
nb_all   = {o: np.array([net_benefit_all(df[o], pt)
                         for pt in thresholds]) for o in outcomes}

fig1, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
for ax, o in zip(axes, outcomes):
    ax.plot(thresholds, nb_model[o], label='Model', color='#185FA5', lw=2)
    ax.plot(thresholds, nb_all[o],   label='Treat all', color='#BA7517',
            lw=1.6, ls='--')
    ax.axhline(0, label='Treat none', color='#5F5E5A', lw=1.4, ls=':')
    ax.set_title(o)
    ax.set_xlabel('Threshold probability (pt)')
    ax.set_ylim(-0.05, max(nb_all[o].max(), nb_model[o].max()) * 1.1 + 0.01)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Net benefit')
fig1.suptitle('Stage 4 - Single-disease decision curves')
fig1.tight_layout()
fig1.savefig(os.path.join(DATA_DIR, "stage4_single_disease_dca.png"), dpi=150)
print("Saved -> stage4_single_disease_dca.png")

# ===============================================================
# STAGE 5 - Naive summation across the three diseases
# ===============================================================
# Add the three single-disease net benefits as if each were a
# separate, independent clinical gain. This is the naive "total value".
nb_naive     = sum(nb_model[o] for o in outcomes)
nb_naive_all = sum(nb_all[o]   for o in outcomes)

# ===============================================================
# STAGE 6 - Patient-level correction
# ===============================================================
# One decision per patient: flagged if flagged for ANY condition;
# a true positive if the patient actually has ANY condition.
risk_cols    = ['risk_' + o for o in outcomes]
has_any      = (df[outcomes] == 1).any(axis=1)          # has >=1 condition
prev_any     = has_any.mean()

nb_patient, nb_patient_all = [], []
for pt in thresholds:
    flag_any = (df[risk_cols] >= pt).any(axis=1)         # flagged for >=1
    tp = np.sum(flag_any & has_any)
    fp = np.sum(flag_any & ~has_any)
    w = pt / (1 - pt)
    nb_patient.append(tp / N - (fp / N) * w)
    # treat-all at patient level: everyone flagged
    nb_patient_all.append(prev_any - (1 - prev_any) * w)
nb_patient     = np.array(nb_patient)
nb_patient_all = np.array(nb_patient_all)

print(f"\nPatient-level prevalence (has any CKM condition): {prev_any*100:.1f}%")

# ---------------------------------------------------------------
# Stage 5 vs Stage 6 comparison figure
# ---------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(thresholds, nb_naive,   label='Naive sum (Stage 5)', color='#A32D2D', lw=2)
ax.plot(thresholds, nb_patient, label='Patient-level (Stage 6)', color='#185FA5', lw=2)
ax.plot(thresholds, nb_naive_all,   color='#A32D2D', lw=1.2, ls='--', alpha=.6,
        label='Naive treat-all')
ax.plot(thresholds, nb_patient_all, color='#185FA5', lw=1.2, ls='--', alpha=.6,
        label='Patient-level treat-all')
ax.axhline(0, color='#5F5E5A', lw=1.2, ls=':')
ax.set_xlabel('Threshold probability (pt)')
ax.set_ylabel('Net benefit')
ax.set_title('Naive summation vs. patient-level net benefit')
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(DATA_DIR, "stage5_6_naive_vs_patient.png"), dpi=150)
print("Saved -> stage5_6_naive_vs_patient.png")

# ===============================================================
# Quantify the overestimation across pt = 0.10 - 0.30
# ===============================================================
band = (thresholds >= 0.10) & (thresholds <= 0.30)
gap_abs = nb_naive[band] - nb_patient[band]
gap_rel = 100 * gap_abs / nb_patient[band]

print("\n" + "=" * 60)
print("OVERESTIMATION OF NET BENEFIT (naive sum vs patient-level)")
print("=" * 60)
print(f"  Threshold band            : pt = 0.10 - 0.30")
print(f"  Mean naive net benefit    : {nb_naive[band].mean():.4f}")
print(f"  Mean patient-level NB      : {nb_patient[band].mean():.4f}")
print(f"  Mean absolute overestimate : {gap_abs.mean():.4f}")
print(f"  Mean relative overestimate : {gap_rel.mean():.1f}%")
print("=" * 60)

# A small table at three representative thresholds
print("\nThreshold-by-threshold:")
print(f"  {'pt':>5} {'naive':>10} {'patient':>10} {'overest %':>10}")
for pt in [0.10, 0.15, 0.20, 0.25, 0.30]:
    i = np.argmin(np.abs(thresholds - pt))
    over = 100 * (nb_naive[i] - nb_patient[i]) / nb_patient[i]
    print(f"  {pt:>5.2f} {nb_naive[i]:>10.4f} {nb_patient[i]:>10.4f} {over:>9.1f}%")

plt.show()
