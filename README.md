# clif_aki_incidence

Weekly incidence of Acute Kidney Injury (AKI) in adults hospitalized at the
University of Chicago Medical Center, 2018-2024, computed from a
[CLIF](https://clif-consortium.github.io/website/) v2.1 database using
[clifpy](https://pypi.org/project/clifpy/).

## What the notebook does

`aki_weekly_incidence.py` is a [marimo](https://marimo.io/) reactive notebook
that:

1. Loads `patient` and `hospitalization` from CLIF and builds an adult cohort
   (age ≥ 18) with `admission_dttm` in 2018-2024.
2. Loads `labs` filtered to `lab_category = "creatinine"` for cohort
   `hospitalization_id`s only.
3. Flags AKI per hospitalization using the KDIGO serum-creatinine criteria:
   - ≥ 0.3 mg/dL absolute rise vs. any prior SCr within a 48-hour window, OR
   - ≥ 1.5× baseline (minimum in-stay SCr) within any 7-day window.
4. Bins admissions by ISO week (Monday-Sunday, local time) and reports weekly
   incidence with a 95% Wilson confidence interval.
5. Stratifies weekly incidence by 10-year age band (18-27, 28-37, ..., 88+).
6. Writes a multi-page PDF report to `results/aki_weekly_incidence_report.pdf`.

## Repo layout

```
aki_weekly_incidence.py    # marimo notebook (the analysis)
pyproject.toml / uv.lock   # dependencies pinned via uv
results/                   # generated PDF report
```

## Running it

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run marimo edit --watch aki_weekly_incidence.py
```

Edit the `DATA_DIR` and `RESULTS_DIR` constants in the config cell to point at
your local CLIF v2.1 database and desired output directory.

## Known limitations / candidate sensitivity analyses

- **Baseline SCr** is defined as the minimum in-stay creatinine. Extending this
  to include a 30-day prior-outpatient window would catch community-acquired
  AKI more reliably.
- **Hospitalizations with no in-stay SCr** are counted as non-AKI in the
  denominator. Restricting to `n_scr > 0` is a reasonable sensitivity analysis.
- **Encounters are not stitched**; readmissions within 6 hours are treated as
  separate. Use `clifpy.utils.stitching_encounters.stitch_encounters` and
  aggregate at `encounter_block` if that matters.
- The **urine-output KDIGO criterion is not applied** (requires reliable hourly
  UOP, typically ICU-only).
- **ESRD / dialysis-dependent patients are not excluded**. To exclude, filter
  `hospital_diagnosis` for ICD codes `N185`, `N186`, `Z992` with
  `present_on_admission = True`.
