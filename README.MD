# Bangladesh CVD Risk Comparison

Repository for the study:

**Comparing Laboratory and Non-Laboratory WHO Models for Estimating Ten-Year Cardiovascular Disease Risk in Low-Resource Primary Care Settings in Bangladesh**

## Contents

- `code/` - Analysis scripts and Streamlit source code used to calculate WHO CVD risk, compare laboratory and non-laboratory models, and generate publication outputs.
- `figures/` - Manuscript figures prepared for the BMJ journal submission.
- `tables/` - Aggregate manuscript tables and summary CSV/XLSX outputs.
- `results/` - Generated aggregate outputs, manuscript drafts, and findings reports.
- `data/` - Placeholder only. Raw, cleaned, imputed, or participant-level datasets are intentionally excluded.

## Overview

This repository contains the code, figures, and tables used to compare laboratory-based and non-laboratory-based WHO cardiovascular disease (CVD) risk prediction models among adults in Bangladesh.

The repository is prepared for journal/publication sharing. It excludes the actual dataset, including raw Portable Health Clinic records, analyzed row-level CSV files, imputed participant-level files, and local environment files.

## Reproducibility

To rerun the analysis, an authorized local copy of the dataset is required. Place approved data files outside version control, for example under `data/raw/`, and update local paths in the analysis scripts as needed.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

The current shared outputs are already included as aggregate tables, summary results, and manuscript figures.

## Data Availability

No participant-level data are included in this repository. See `DATA_AVAILABILITY.md` for the repository-level data sharing note to adapt for the final BMJ submission.

## Citation

If you use this repository, please cite the associated journal article.

## License

MIT License
