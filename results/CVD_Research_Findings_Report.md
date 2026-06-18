# Comparing Laboratory and Non-Laboratory WHO CVD Risk Models in Bangladesh

**Target venue:** Peer-reviewed journal  
**Source dataset:** `cvd/resource/analyzed/v2.30`  


## Executive Summary

This report uses the v2.30 analytic dataset to compare the 2019 WHO laboratory and non-laboratory cardiovascular disease (CVD) risk charts among adults screened through the Portable Health Clinic program in Bangladesh. Participant-level data are not included in the publication repository.

Key v2.30 findings:
- WHO non-laboratory domain cohort: **14,085** adults aged 40-74 years.
- Paired laboratory/non-laboratory cohort: **1,762** adults with complete inputs for both charts.
- Mean 10-year risk in the paired cohort was **7.51%** with the non-laboratory chart and **9.70%** with the laboratory chart.
- The mean paired bias was **2.19 percentage points** for laboratory minus non-laboratory risk.
- Spearman correlation was **rho = 0.883** and quadratic weighted Cohen kappa was **0.767**.
- At the >=20% threshold, the non-laboratory chart identified **49** high-risk adults versus **177** by the laboratory chart and missed **137/177 (77.4%)** laboratory-defined high-risk adults.

---

## Cohort Definitions

| Cohort | File | N | Description |
| :--- | :--- | :---: | :--- |
| **General non-lab cohort** | `cvd_nonlab.csv` | 35,768 | All records with non-laboratory processing fields. |
| **General lab cohort** | `cvd_lab.csv` | 3,241 | Records with laboratory fields available. |
| **WHO non-lab domain cohort** | `cvd_who_nonlab_domain.csv` | 14,085 | Adults aged 40-74 years within WHO non-lab chart domains. |
| **WHO lab/paired domain cohort** | `cvd_who_lab_domain.csv` / `cvd_paired.csv` | 1,762 | Adults with complete paired lab and non-lab WHO risk estimates. |

---

## Table 1: Baseline Characteristics: Comprehensive Cohort Analysis

| Variable | General Cohort Non-Lab (N=35,768) | General Cohort Lab (N=3,241) | WHO Domain Cohort Non-Lab (N=14,085) | WHO Domain Cohort Lab (N=1,762) |
| :--- | :---: | :---: | :---: | :---: |
| **Study sample** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Total N | 35,768 | 3,241 | 14,085 | 1,762 |
| **Age** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Mean ± SD (years) | 38.1 ± 14.4 | 43.8 ± 15.3 | 51.4 ± 8.7 | 53.1 ± 8.7 |
| &nbsp;&nbsp;&nbsp;&nbsp;Age ≥ 60 yr, n (%) | 3,731 (10.4%) | 550 (17.0%) | 3,190 (22.6%) | 457 (25.9%) |
| **Age distribution** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;< 40 | 21,141 (59.1%) | 1,386 (42.8%) | 0 (0.0%) | 0 (0.0%) |
| &nbsp;&nbsp;&nbsp;&nbsp;40–44 | 3,453 (9.7%) | 327 (10.1%) | 3,453 (24.5%) | 327 (18.6%) |
| &nbsp;&nbsp;&nbsp;&nbsp;45–49 | 3,001 (8.4%) | 402 (12.4%) | 3,000 (21.3%) | 402 (22.8%) |
| &nbsp;&nbsp;&nbsp;&nbsp;50–54 | 2,547 (7.1%) | 277 (8.5%) | 2,547 (18.1%) | 277 (15.7%) |
| &nbsp;&nbsp;&nbsp;&nbsp;55–59 | 1,895 (5.3%) | 299 (9.2%) | 1,895 (13.5%) | 299 (17.0%) |
| &nbsp;&nbsp;&nbsp;&nbsp;60–64 | 1,619 (4.5%) | 224 (6.9%) | 1,619 (11.5%) | 224 (12.7%) |
| &nbsp;&nbsp;&nbsp;&nbsp;65–69 | 935 (2.6%) | 166 (5.1%) | 935 (6.6%) | 166 (9.4%) |
| &nbsp;&nbsp;&nbsp;&nbsp;70–74 | 637 (1.8%) | 67 (2.1%) | 636 (4.5%) | 67 (3.8%) |
| &nbsp;&nbsp;&nbsp;&nbsp;≥ 75 | 540 (1.5%) | 93 (2.9%) | 0 (0.0%) | 0 (0.0%) |
| **Sex** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Male | 18,938 (52.9%) | 1,825 (56.3%) | 7,580 (53.8%) | 975 (55.3%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Female | 16,830 (47.1%) | 1,416 (43.7%) | 6,505 (46.2%) | 787 (44.7%) |
| **Vitals** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;SBP (mmHg), mean ± SD | 121.3 ± 18.6 | 124.6 ± 17.5 | 128.5 ± 20.6 | 128.6 ± 17.7 |
| &nbsp;&nbsp;&nbsp;&nbsp;BMI (kg/m²), mean ± SD | 22.9 ± 4.6 | 26.1 ± 4.9 | 23.8 ± 5.2 | 26.7 ± 4.6 |
| **Risk factors** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Current smoker | 6,082 (17.0%) | 1,715 (52.9%) | 3,369 (23.9%) | 875 (49.7%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Diabetes | 8,591 (24.0%) | 1,106 (34.1%) | 5,230 (37.1%) | 795 (45.1%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Total cholesterol (mmol/L), mean ± SD | 5.53 ± 1.30 | 5.53 ± 1.30 | 5.67 ± 1.16 | 5.67 ± 1.16 |
| &nbsp;&nbsp;&nbsp;&nbsp;(Available N for cholesterol) | (n=3,241) | (n=3,241) | (n=1,762) | (n=1,762) |
| **Smoking status** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Smoker | 6,082 (17.0%) | 1,715 (52.9%) | 3,369 (23.9%) | 875 (49.7%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Non-smoker | 29,686 (83.0%) | 1,526 (47.1%) | 10,716 (76.1%) | 887 (50.3%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Missing | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Blood pressure category** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Normal | 13,955 (39.0%) | 901 (27.8%) | 3,635 (25.8%) | 381 (21.6%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Elevated | 2,790 (7.8%) | 274 (8.5%) | 1,091 (7.7%) | 168 (9.5%) |
| &nbsp;&nbsp;&nbsp;&nbsp;HTN Stage 1 | 12,715 (35.5%) | 1,397 (43.1%) | 5,599 (39.8%) | 790 (44.8%) |
| &nbsp;&nbsp;&nbsp;&nbsp;HTN Stage 2 | 4,465 (12.5%) | 460 (14.2%) | 2,837 (20.1%) | 314 (17.8%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Hypertensive crisis | 73 (0.2%) | 11 (0.3%) | 53 (0.4%) | 6 (0.3%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Missing | 1,770 (4.9%) | 198 (6.1%) | 870 (6.2%) | 103 (5.8%) |
| **Location type** | | | | |
| &nbsp;&nbsp;&nbsp;&nbsp;Urban | 10,331 (28.9%) | 1,756 (54.2%) | 3,668 (26.0%) | 1,082 (61.4%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Rural | 16,948 (47.4%) | 873 (26.9%) | 9,285 (65.9%) | 472 (26.8%) |
| &nbsp;&nbsp;&nbsp;&nbsp;Semi-urban | 7,787 (21.8%) | 345 (10.6%) | 806 (5.7%) | 67 (3.8%) |

---

## Section 1: Baseline CVD Risk Burden in the WHO Non-Lab Domain Cohort

The WHO non-laboratory domain cohort included **14,085** adults. Mean non-laboratory 10-year CVD risk was **6.08%** (SD **4.61**), with a median of **5.0%**.

### Overall Non-Lab Risk Distribution

| Risk Category | N | % |
| :--- | :---: | :---: |
| **<5%** | 6,821 | 48.4% |
| **5% to <10%** | 4,520 | 32.1% |
| **10% to <20%** | 2,492 | 17.7% |
| **20% to <30%** | 237 | 1.7% |
| **≥30%** | 15 | 0.1% |

### Age Gradient

| Age Band | N | ≥10% | ≥20% | Male N | Female N |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **40-44** | 3,453 | 0.7% | 0.0% | 1,712 | 1,741 |
| **45-49** | 3,000 | 1.6% | 0.1% | 1,510 | 1,490 |
| **50-54** | 2,547 | 8.5% | 0.2% | 1,466 | 1,081 |
| **55-59** | 1,895 | 28.3% | 1.1% | 1,204 | 691 |
| **60-64** | 1,619 | 40.0% | 1.5% | 855 | 764 |
| **65-69** | 935 | 68.1% | 6.1% | 519 | 416 |
| **70-74** | 636 | 100.0% | 22.5% | 314 | 322 |

Risk ≥10% increased from **0.7%** in adults aged 40-44 years to **100.0%** in adults aged 70-74 years.

### Location Pattern

| Location Type | N | Mean Risk | ≥10% n (%) | ≥20% n (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Rural** | 9,285 | 5.96% | 1,724 (18.6%) | 159 (1.7%) |
| **Semi-urban** | 806 | 4.99% | 102 (12.7%) | 17 (2.1%) |
| **Unknown** | 326 | 7.38% | 89 (27.3%) | 15 (4.6%) |
| **Urban** | 3,668 | 6.52% | 829 (22.6%) | 61 (1.7%) |

---

## Section 2: Paired Agreement Between Laboratory and Non-Laboratory WHO Charts

The paired cohort included **1,762** adults. Laboratory estimates were higher than non-laboratory estimates in **937 (53.2%)**, equal in **503 (28.5%)**, and lower in **322 (18.3%)**.

### Continuous Agreement

| Metric | Value |
| :--- | :--- |
| **Mean non-lab risk** | 7.51% |
| **Mean lab risk** | 9.70% |
| **Mean bias, lab - non-lab** | 2.19 percentage points |
| **Median bias, lab - non-lab** | 1.0 percentage points |
| **95% limits of agreement** | -4.71 to 9.09 percentage points |
| **Pearson r** | 0.872 (p <0.001) |
| **Spearman rho** | 0.883 (p <0.001) |
| **Wilcoxon signed-rank p** | <0.001 |

### Categorical Agreement

| Non-lab Category / Lab Category | <5% | 5% to <10% | 10% to <20% | 20% to <30% | ≥30% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **<5%** | 400 | 151 | 9 | 0 | 0 |
| **5% to <10%** | 44 | 399 | 257 | 4 | 0 |
| **10% to <20%** | 0 | 32 | 284 | 124 | 9 |
| **20% to <30%** | 0 | 0 | 9 | 23 | 11 |
| **≥30%** | 0 | 0 | 0 | 1 | 5 |

Exact agreement was **1,111/1,762 (63.1%)** and within-one-category agreement was **1,740/1,762 (98.8%)**. Cohen kappa was **0.476** unweighted, **0.623** with linear weights, and **0.767** with quadratic weights.

---

## Table 2: Clinical classification performance at key thresholds

| Threshold | TP | FN | FP | TN | Sensitivity % (95% CI) | Specificity % (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lab ≥ 20% vs NL ≥ 20%** | 40 | 137 | 9 | 1,576 | 22.6 (17.1–29.3) | 99.4 (98.9–99.7) |
| **Lab ≥ 10% vs NL ≥ 10%** | 466 | 270 | 32 | 994 | 63.3 (59.8–66.7) | 96.9 (95.6–97.8) |
| **Lab ≥ 20% vs NL ≥ 10%\*** | 173 | 4 | 325 | 1,260 | 97.7 (94.3–99.1) | 79.5 (77.4–81.4) |

*\*Note: Strategy of using non-laboratory threshold of ≥10% to screen for laboratory-based pharmacotherapy threshold of ≥20%.*

At the ≥20% pharmacotherapy threshold, sensitivity was **22.6%** and specificity was **99.4%**. The non-laboratory chart missed **137** of **177** laboratory-defined high-risk adults.

---

## SUPPLEMENTARY TABLES

### Table S1: Distribution of 10-year CVD risk by sex using laboratory and non-laboratory models

| Risk Category | WHO Non-Laboratory Model: Total n (%) | WHO Non-Laboratory Model: Male | WHO Non-Laboratory Model: Female | WHO Laboratory Model: Total n (%) | WHO Laboratory Model: Male | WHO Laboratory Model: Female |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Very low (<5%)** | 6,821 (48.4%) | 2,731 (36.0%) | 4,090 (62.9%) | 444 (25.2%) | 101 (10.4%) | 343 (43.6%) |
| **Low (5–<10%)** | 4,520 (32.1%) | 2,775 (36.6%) | 1,745 (26.8%) | 582 (33.0%) | 338 (34.7%) | 244 (31.0%) |
| **Moderate (10–<20%)** | 2,492 (17.7%) | 1,844 (24.3%) | 648 (10.0%) | 559 (31.7%) | 386 (39.6%) | 173 (22.0%) |
| **High (20–<30%)** | 237 (1.7%) | 215 (2.8%) | 22 (0.3%) | 152 (8.6%) | 126 (12.9%) | 26 (3.3%) |
| **Very high (≥30%)** | 15 (0.1%) | 15 (0.2%) | 0 (0.0%) | 25 (1.4%) | 24 (2.5%) | 1 (0.1%) |

### Table S2: Bias between non-laboratory and laboratory risk scores, stratified by laboratory risk band

| Lab Risk Band | n | Mean bias (pp) | SD (pp) | Underestimated (%) |
| :--- | :---: | :---: | :---: | :---: |
| **<5%** | 444 | 0.19 | 0.65 | 7.2% |
| **5–<10%** | 582 | -0.99 | 2.05 | 51.9% |
| **10–<20%** | 559 | -3.52 | 3.17 | 78.7% |
| **20–<30%** | 152 | -7.31 | 3.86 | 91.4% |
| **≥30%** | 25 | -11.40 | 6.10 | 96.0% |
| **Overall** | 1,762 | -2.19 | 3.52 | 53.2% |

### Table S3: Clinical profile of missed vs. detected high-risk individuals

| Characteristic | Missed (n=137) | Detected (n=40) |
| :--- | :---: | :---: |
| **Mean age (years)** | 67.4 | 73.0 |
| **Mean SBP (mmHg)** | 143.6 | 174.3 |
| **Mean BMI (kg/m²)** | 23.4 | 25.9 |
| **Primary risk driver** | Cholesterol/diabetes | Extreme SBP/BMI |

---

## Interpretation for Manuscript

The v2.30 dataset supports the main manuscript conclusion: the non-laboratory WHO chart preserves rank ordering reasonably well but materially under-identifies adults who cross clinically actionable laboratory-based risk thresholds. The discrepancy is most important at the >=20% threshold, where laboratory-based assessment identifies **177 (10.0%)** high-risk adults compared with **49 (2.8%)** by the non-laboratory chart.

## Reproducibility Note

All values in this report were regenerated from `cvd/resource/analyzed/v2.30`. The publication repository contains only aggregate outputs, figures, tables, and code; it does not contain participant-level data.
