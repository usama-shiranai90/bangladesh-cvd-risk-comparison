# BMJ HCI Journal Figures and Tables Plan

**Source dataset:** `cvd/resource/analyzed/v2.30`  
**Study:** Comparing laboratory and non-laboratory WHO models for estimating ten-year cardiovascular disease risk in low-resource primary care settings in Bangladesh.


---

## Display Item Plan

| Item | File(s) | Purpose | Key v2.30 values |
| :--- | :--- | :--- | :--- |
| **Table 1** | `tables/table1_baseline.csv` | Baseline characteristics | WHO non-lab N=14,085; paired/lab N=1,762 |
| **Table 2** | `tables/table2_threshold_performance.csv` | Clinical threshold performance | Sensitivity 22.6% vs 63.3% vs 97.7% |
| **Figure 1** | `figures/Fig1_study_flow.svg` / `.pdf` | Cohort flow diagram | General non-lab 35,768; general lab 3,241; paired 1,762 |
| **Figure 2** | `figures/Fig2_risk_distribution.svg` / `.pdf` | Risk distribution by model | Non-lab ≥20% 49; lab ≥20% 177 |
| **Figure 3** | `figures/Fig3_agreement.svg` / `.pdf` | Agreement and Bland-Altman | Exact agreement 63.1%; quadratic kappa 0.767 |
| **Figure 4** | `figures/Fig4_age_divergence.svg` / `.pdf` | Age-stratified divergence | Non-lab ≥10% reaches 100.0% at age 70-74 |
| **Figure 5** | `figures/Fig5_clinical_utility.svg` / `.pdf` | Missed high-risk & decision utility | ≥20% sensitivity 22.6%; missed 77.4% |
| **Figure 6** | `figures/Fig6_sex_age_heatmap.svg` / `.pdf` | Sex and age heterogeneity | Male paired bias 2.84; female bias 1.39 |

---

## Recommended Main Figures

### Figure 1. Study Flow
* **File:** `figures/Fig1_study_flow.svg` and `figures/Fig1_study_flow.pdf`
* **Legend:** The v2.30 analytic dataset contained **35,768** records in the general non-laboratory cohort and **3,241** records in the general laboratory cohort. After applying WHO chart age and domain restrictions, **14,085** adults remained in the non-laboratory WHO domain cohort and **1,762** adults remained in the paired laboratory/non-laboratory comparison cohort.

### Figure 2. Risk Distribution by WHO Model
* **File:** `figures/Fig2_risk_distribution.svg` and `figures/Fig2_risk_distribution.pdf`
* **Legend:** Non-laboratory WHO risk in the domain cohort (N=14,085) had mean **6.08%** and median **5.0%**. In the paired cohort (N=1,762), mean risk was **7.51%** using the non-laboratory chart and **9.70%** using the laboratory chart. At the ≥20% threshold, the non-laboratory chart classified **49 (2.8%)** adults as high risk compared with **177 (10.0%)** using the laboratory chart.

### Figure 3. Agreement and Reclassification
* **File:** `figures/Fig3_agreement.svg`, `figures/Fig3_1_bland_altman.svg`, and `figures/Fig3_2_concordance.svg`
* **Legend:** Among **1,762** paired observations, exact five-band categorical agreement was **63.1%**, within-one-category agreement was **98.8%**, and quadratic weighted Cohen kappa was **0.767**. Spearman correlation was **0.883**. Mean bias (laboratory minus non-laboratory risk) was **2.19 percentage points**, with 95% limits of agreement from **-4.71** to **9.09** percentage points.

### Figure 4. Age-Stratified Risk Divergence
* **File:** `figures/Fig4_age_divergence.svg` and `figures/Fig4_age_divergence.pdf`
* **Legend:** In the WHO non-laboratory domain cohort, risk ≥10% increased from **0.7%** at age 40-44 years to **100.0%** at age 70-74 years. Risk ≥20% increased from **0.0%** to **22.5%** across the same age bands.

### Figure 5. Missed High-Risk and Clinical Utility
* **File:** `figures/Fig5_clinical_utility.svg` and `figures/Fig5_clinical_utility.pdf`
* **Legend:** At the ≥20% treatment threshold, the laboratory chart identified **177** high-risk adults and the non-laboratory chart identified **49**. The non-laboratory chart missed **137 (77.4%)** laboratory-defined high-risk adults, with sensitivity **22.6%**, specificity **99.4%**, PPV **81.6%**, and NPV **92.0%**.

### Figure 6. Sex and Age Heterogeneity
* **File:** `figures/Fig6_sex_age_heatmap.svg` and `figures/Fig6_sex_age_heatmap.pdf`
* **Legend:** The paired cohort included **975 men** and **787 women**. Mean laboratory-minus-non-laboratory bias was **2.84 percentage points** in men and **1.39 percentage points** in women. Non-laboratory risk ≥10% in the WHO domain cohort rose consistently with age in both sexes.

---

## Main Tables

### Table 1. Baseline Characteristics: Comprehensive Cohort Analysis
* **File:** `tables/table1_baseline.csv` (CSV version)
* **Description:** Provides the demographical, vitals, and clinical characteristic differences between the General Cohort (Non-Lab and Lab subsets) and the WHO Domain Cohorts.

### Table 2. Clinical classification performance at key thresholds
* **File:** `tables/table2_threshold_performance.csv` (CSV version)
* **Data:**
| Threshold | TP | FN | FP | TN | Sensitivity % (95% CI) | Specificity % (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lab ≥ 20% vs NL ≥ 20%** | 40 | 137 | 9 | 1,576 | 22.6 (17.1–29.3) | 99.4 (98.9–99.7) |
| **Lab ≥ 10% vs NL ≥ 10%** | 466 | 270 | 32 | 994 | 63.3 (59.8–66.7) | 96.9 (95.6–97.8) |
| **Lab ≥ 20% vs NL ≥ 10%\*** | 173 | 4 | 325 | 1,260 | 97.7 (94.3–99.1) | 79.5 (77.4–81.4) |

*\*Note: Strategy of using non-laboratory threshold of ≥10% to screen for laboratory-based pharmacotherapy threshold of ≥20%.*

---

## Supplementary Tables

### Table S1: Distribution of 10-year CVD risk by sex using laboratory and non-laboratory models
* **File:** `tables/table_s1_risk_distribution.csv`
* **Data:**
| Risk category | Non-lab male | Non-lab female | Lab male | Lab female |
| :--- | :---: | :---: | :---: | :---: |
| **<5%** | 165 (16.9%) | 395 (50.2%) | 101 (10.4%) | 343 (43.6%) |
| **5% to <10%** | 417 (42.8%) | 287 (36.5%) | 338 (34.7%) | 244 (31.0%) |
| **10% to <20%** | 346 (35.5%) | 103 (13.1%) | 386 (39.6%) | 173 (22.0%) |
| **20% to <30%** | 41 (4.2%) | 2 (0.3%) | 126 (12.9%) | 26 (3.3%) |
| **≥30%** | 6 (0.6%) | 0 (0.0%) | 24 (2.5%) | 1 (0.1%) |

### Table S2: Bias between non-laboratory and laboratory risk scores, stratified by laboratory risk band
* **File:** `tables/table_s2_bias.csv`
* **Data:**
| Lab risk band | n | Mean bias (pp) | SD (pp) | Underestimated (%) |
| :--- | :---: | :---: | :---: | :---: |
| **<5%** | 444 | 0.19 | 0.65 | 7.2% |
| **5–<10%** | 582 | -0.99 | 2.05 | 51.9% |
| **10–<20%** | 559 | -3.52 | 3.17 | 78.7% |
| **20–<30%** | 152 | -7.31 | 3.86 | 91.4% |
| **≥30%** | 25 | -11.40 | 6.10 | 96.0% |
| **Overall** | 1,762 | -2.19 | 3.52 | 53.2% |

### Table S3: Clinical profile of missed vs. detected high-risk individuals
* **File:** `tables/table_s3_missed_vs_detected.csv`
* **Data:**
| Characteristic | Missed (n=137) | Detected (n=40) |
| :--- | :---: | :---: |
| **Mean age (years)** | 67.4 | 73.0 |
| **Mean SBP (mmHg)** | 143.6 | 174.3 |
| **Mean BMI (kg/m²)** | 23.4 | 25.9 |
| **Primary risk driver** | Cholesterol/diabetes | Extreme SBP/BMI |

---

## Source Mapping

| Display item | Primary code | Primary data |
| :--- | :--- | :--- |
| **Table 1** | `code/streamlit/utils/stats_summary.py` | `cvd/resource/analyzed/v2.30/*.csv` |
| **Table 2** | `code/streamlit/utils/stats_summary.py` | `cvd/resource/analyzed/v2.30/*.csv` |
| **Figures 1-6** | `code/streamlit/modules/journal_figures.py` | `cvd/resource/analyzed/v2.30/*.csv` |
| **Agreement metrics** | `code/streamlit/utils/risk_engines.py` | `cvd_paired.csv` |

---

## Quality Check

- Values in this file were validated against the v2.30 analytic dataset.
- No participant-level dataset was copied into the BMJ publication folder.
- Stale placeholder values and older cohort numbers have been replaced.

