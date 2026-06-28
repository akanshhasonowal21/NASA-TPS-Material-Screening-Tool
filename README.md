# 🚀 TPS Material Intelligence — Two-Stage ML Pipeline

### NASA Thermal Protection System Material Screening Tool

**Project Context:** NASA mission design requires rapid screening of Thermal Protection System (TPS) candidate materials before full experimental characterisation. This project demonstrates a data-driven screening tool that predicts missing thermal conductivity values and ranks materials given mission constraints (temperature, budget, mission type).

---

## 🏗️ Architecture: Two-Stage Pipeline

1. **Stage 1: Thermal Conductivity Prediction** via Group-K-Fold Cross-Validation
2. **Stage 2: Mission-Aware Physics-Informed Material Ranking** This pipeline structure ensures that the machine learning output (Stage 1) directly feeds into the deterministic scoring and ranking system (Stage 2) — representing a real-world engineering pipeline architecture.

---

## 📊 Dataset Construction

The dataset is compiled from published NASA technical reports, ESA documentation, and peer-reviewed TPS literature. Each row represents a material measured at a specific temperature and state (virgin = pre-ablation, char = post-ablation).

### Features

* `density` (kg/m³) — Bulk density
* `specific_heat` (J/kg·K) — Thermal energy storage capacity (Cp)
* `emissivity` (dimensionless, 0–1) — Radiative surface property
* `temperature` (K) — Measurement temperature
* `material_class` — Ablator / ceramic_tile / blanket / RCC
* `state` — Virgin or char

*(Note: An anomalous data entry for 'advanced Carbon Carbon composite' with a specific heat of 1.2 J/kg·K against a dataset mean of ~1000 J/kg·K was identified and handled).*

---

## 🔬 Stage 1 — Thermal Conductivity Prediction

**Target:** `thermal_conductivity` (W/m·K) — some virgin-state entries are missing and are predicted by the model.

### Feature Engineering (Physics-Informed)

To keep the model physically driven rather than relying on category lookups, `material_class` encoding was deliberately excluded. `thermal_diffusivity` and `density_temp` ($\rho$/T) were also excluded to prevent data leakage and maintain physical interpretability.

* `temperature`: Primary driver of thermal conductivity (Debye model)
* `temperature_sq`: Captures nonlinear k-T relationship (quadratic regime)
* `density`: Material structural property
* `specific_heat`: Thermal energy storage capacity
* `vol_heat_cap`: Density × specific heat ($\rho$ × Cp), representing volumetric heat capacity
* `emissivity`: Radiative surface property, correlated with material class

### Validation Strategy

**Group K-Fold Cross-Validation:** Each fold holds out all rows from specific unique materials rather than random rows. This prevents material-level data leakage and is the only valid evaluation strategy for repeated-measurement data. Models evaluated include Ridge Regression, Random Forest, and XGBoost.

### Scope Definition & High-k Material Exclusion

RCC (Reinforced Carbon-Carbon) and structurally similar carbon composites are **excluded** from Stage 1 ML prediction for scientifically quantified reasons:

1. **Conductivity Regime Gap:** RCC thermal conductivity ranges from 9.5–12.5 W/m·K, while insulating TPS materials range from 0.02–0.50 W/m·K (~52× difference).
2. **Feature Space Isolation:** RCC density is 1630 kg/m³ compared to 130–530 kg/m³ for insulating materials.
3. **Statistical Validity:** With only 5 RCC data points and no similar materials in training, Group K-Fold produces unbounded negative R² scores (extrapolation failure).

**Scope:** The ML model applies exclusively to insulating TPS materials (ablators, ceramic tiles, fibrous blankets, surface coatings). RCC is retained in Stage 2 with its known measured properties.

---

## 🏆 Stage 2 — Mission-Aware Scoring & Ranking

Materials are evaluated using an interpretable, physics-informed weighted scoring system rather than a black-box model.

### Scoring Formula

```text
Score = (w1 × ThermalScore) + (w2 × TemperatureScore) + (w3 × CostScore) + (w4 × InstallationScore)

```

**Metrics:**

* **ThermalScore:** Normalized inverse of *mean* thermal conductivity (lower k → better insulation)
* **TemperatureScore:** Safety margin defined as `(max_use_temp − mission_temp) / mission_temp`, clamped at [0,1]
* **CostScore:** Normalized inverse of cost per m² relative to budget
* **InstallationScore:** Normalized inverse of installation time

**Mission Weight Profiles:**

| Mission Type | w1 (Thermal) | w2 (Temp) | w3 (Cost) | w4 (Install) |
| --- | --- | --- | --- | --- |
| **Single-use ablative** | 0.40 | 0.40 | 0.15 | 0.05 |
| **Multi-use reusable** | 0.30 | 0.25 | 0.20 | 0.25 |
| **Budget constrained** | 0.20 | 0.30 | 0.40 | 0.10 |

---

## 🧠 Explainability

**SHAP (SHapley Additive exPlanations)** is utilized to decompose each prediction into per-feature contributions. This validates which physical properties drive thermal conductivity in the trained model. TreeExplainer is used for ensemble models (XGBoost/RF) and KernelExplainer is used for Ridge Regression.

---

## 🎯 Results Summary & Measurable Objectives

| Objective | Metric | Threshold | Status |
| --- | --- | --- | --- |
| Stage 1: Predict thermal conductivity | RMSE, R², Group K-Fold CV | R² > 0.85 on held-out groups | Completed |
| Stage 1: Compare models | ΔRMSE between XGBoost, RF, Ridge | XGBoost or RF should win | Completed |
| Stage 2: Ranking consistency | Manual validation vs NASA docs | Ablators > Tiles > Blankets @ 1800K | ✅ Validated |
| Explainability | SHAP mean absolute values | Top 3 features identified | ✅ Completed |
| Data completeness | % missing k values filled | 100% resolved | ✅ Completed |

---

## 📁 Repository Files

* `eda_overview.png` — EDA visualisations
* `stage1_cv_results.png` — Model comparison
* `stage1_fit.png` — Predicted vs Actual + Residuals
* `stage2_rankings.png` — Material rankings by mission type
* `stage2_decomposition.png` — Score breakdown
* `shap_summary.png` — SHAP beeswarm plot
* `shap_bar.png` — Feature importance bar chart
* `tps_app.py` — Interactive Streamlit web app code export