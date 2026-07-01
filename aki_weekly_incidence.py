import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Weekly Incidence of AKI in Adults Hospitalized at UCMC (2018–2024)

    **Data source:** CLIF v2.1 (UCMC)
    **AKI definition:** KDIGO Serum Creatinine criteria
    - ≥0.3 mg/dL absolute rise vs any SCr within the prior 48h, OR
    - ≥1.5× baseline (minimum SCr during hospitalization) within any 7-day window

    **Cohort:** adults (age ≥18) with `admission_dttm` between 2018-01-01 and 2024-12-31.
    **Denominator per week:** count of adult admissions in that ISO week.
    **Numerator per week:** count of those admissions that developed AKI at any point during the stay.
    """)
    return


@app.cell
def _():
    # --- Configuration ---
    DATA_DIR = "/Users/williamparker/Desktop/active projects/data/CLIF_databases/UCMC_CLIF_v2_1"
    RESULTS_DIR = "/Users/williamparker/Desktop/active projects/projects/incidence_of_AKI/results"
    FILETYPE = "parquet"
    TIMEZONE = "US/Central"  # UChicago Medicine
    STUDY_START = "2018-01-01"
    STUDY_END = "2024-12-31"
    MIN_ADULT_AGE = 18

    import pandas as pd
    import numpy as np

    return DATA_DIR, FILETYPE, MIN_ADULT_AGE, RESULTS_DIR, TIMEZONE, np, pd


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE):
    # --- Load patient + hospitalization via clifpy individual table classes ---
    from clifpy.tables import Patient, Hospitalization

    patient = Patient.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    )
    hospitalization = Hospitalization.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    )

    patient_df = patient.df
    hosp_df = hospitalization.df
    print(f"patients: {len(patient_df):,}   hospitalizations: {len(hosp_df):,}")
    return (hosp_df,)


@app.cell
def _(MIN_ADULT_AGE, hosp_df):
    # --- Build adult cohort admitted in the study window (2018-2024 by calendar year) ---
    cohort_hosp = hosp_df[
        (hosp_df["age_at_admission"] >= MIN_ADULT_AGE)
        & hosp_df["age_at_admission"].notna()
        & (hosp_df["admission_dttm"].dt.year >= 2018)
        & (hosp_df["admission_dttm"].dt.year <= 2024)
    ][
        [
            "patient_id",
            "hospitalization_id",
            "admission_dttm",
            "discharge_dttm",
            "age_at_admission",
        ]
    ].copy()

    print(f"Adult hospitalizations 2018-2024: {len(cohort_hosp):,}")
    print(f"Unique patients: {cohort_hosp['patient_id'].nunique():,}")
    cohort_hosp.head()
    return (cohort_hosp,)


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE, cohort_hosp):
    # --- Load creatinine labs for cohort only ---
    from clifpy.tables import Labs

    hosp_ids = cohort_hosp["hospitalization_id"].unique().tolist()

    labs = Labs.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": hosp_ids,
            "lab_category": ["creatinine"],
        },
        columns=[
            "hospitalization_id",
            "lab_category",
            "lab_collect_dttm",
            "lab_value_numeric",
        ],
    )
    scr = labs.df
    print(f"Creatinine rows loaded: {len(scr):,}")
    scr.head()
    return (scr,)


@app.cell
def _(cohort_hosp, scr):
    # --- Clean creatinine values and restrict to values during hospitalization ---
    scr_clean = scr.dropna(subset=["lab_value_numeric", "lab_collect_dttm"]).copy()

    # Physiologic sanity bounds (drop obvious errors: <0.1 or >30 mg/dL)
    scr_clean = scr_clean[
        (scr_clean["lab_value_numeric"] >= 0.1)
        & (scr_clean["lab_value_numeric"] <= 30)
    ]

    scr_clean = scr_clean.merge(
        cohort_hosp[["hospitalization_id", "admission_dttm", "discharge_dttm"]],
        on="hospitalization_id",
        how="inner",
    )

    # Keep only values collected during the stay (admission_dttm <= collect <= discharge_dttm)
    scr_clean = scr_clean[
        (scr_clean["lab_collect_dttm"] >= scr_clean["admission_dttm"])
        & (
            scr_clean["lab_collect_dttm"]
            <= scr_clean["discharge_dttm"].fillna(scr_clean["lab_collect_dttm"])
        )
    ]

    scr_clean = scr_clean.sort_values(["hospitalization_id", "lab_collect_dttm"])
    print(
        f"Cleaned in-stay creatinine rows: {len(scr_clean):,} "
        f"across {scr_clean['hospitalization_id'].nunique():,} hospitalizations"
    )
    return (scr_clean,)


@app.cell
def _(np, pd, scr_clean):
    # --- Detect KDIGO AKI per hospitalization ---
    # Per KDIGO SCr criteria:
    #   (A) SCr increase >= 0.3 mg/dL from any prior SCr in past 48h, OR
    #   (B) SCr >= 1.5x baseline within any 7-day window
    # Baseline := minimum SCr observed during the hospitalization (standard when
    # outpatient baseline is unavailable). This is a conservative operationalization
    # commonly used in hospital-incident AKI studies.

    def flag_aki(group: pd.DataFrame) -> pd.Series:
        g = group.set_index("lab_collect_dttm")["lab_value_numeric"].sort_index()

        # (A) 48h absolute rise: current value - min value in past 48h >= 0.3
        min_48h = g.rolling("48h", closed="both").min()
        abs_rise = (g - min_48h) >= 0.3

        # (B) 7d ratio: current value / min value in past 7d >= 1.5
        min_7d = g.rolling("7D", closed="both").min()
        ratio_rise = (g / min_7d) >= 1.5

        aki_event = abs_rise | ratio_rise

        if aki_event.any():
            first_aki_time = aki_event[aki_event].index.min()
            first_aki_time_from_adm = (
                first_aki_time - group["admission_dttm"].iloc[0]
            ).total_seconds() / 3600.0
        else:
            first_aki_time = pd.NaT
            first_aki_time_from_adm = np.nan

        return pd.Series(
            {
                "n_scr": len(g),
                "baseline_scr": g.min(),
                "peak_scr": g.max(),
                "aki": bool(aki_event.any()),
                "first_aki_dttm": first_aki_time,
                "hours_to_first_aki": first_aki_time_from_adm,
            }
        )

    aki_by_hosp = (
        scr_clean.groupby("hospitalization_id", group_keys=False)
        .apply(flag_aki)
        .reset_index()
    )
    print(
        f"Hospitalizations with any SCr: {len(aki_by_hosp):,}   "
        f"with AKI: {aki_by_hosp['aki'].sum():,} "
        f"({aki_by_hosp['aki'].mean() * 100:.1f}%)"
    )
    aki_by_hosp.head()
    return (aki_by_hosp,)


@app.cell
def _(aki_by_hosp, cohort_hosp):
    # --- Attach AKI flag to every cohort hospitalization ---
    # Hospitalizations with no in-stay SCr are treated as AKI = False (cannot detect).
    cohort_aki = cohort_hosp.merge(
        aki_by_hosp[["hospitalization_id", "n_scr", "aki", "first_aki_dttm"]],
        on="hospitalization_id",
        how="left",
    )
    cohort_aki["aki"] = cohort_aki["aki"].fillna(False).astype(bool)
    cohort_aki["n_scr"] = cohort_aki["n_scr"].fillna(0).astype(int)

    total = len(cohort_aki)
    with_scr = (cohort_aki["n_scr"] > 0).sum()
    with_aki = cohort_aki["aki"].sum()
    print(f"Cohort total:                {total:,}")
    print(
        f"With any in-stay SCr:        {with_scr:,} ({with_scr / total * 100:.1f}%)"
    )
    print(f"With detected AKI (KDIGO):   {with_aki:,} ({with_aki / total * 100:.1f}%)")
    return (cohort_aki,)


@app.cell
def _(cohort_aki):
    # --- Weekly incidence (denominator = admissions per ISO week) ---
    df = cohort_aki.copy()
    # Convert admission timestamp to local time before taking week start,
    # so weeks reflect local calendar.
    df["admission_local"] = df["admission_dttm"].dt.tz_convert("US/Central")
    # Week start = Monday of the ISO week containing the admission
    df["week_start"] = (
        df["admission_local"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    )

    weekly = (
        df.groupby("week_start")
        .agg(
            admissions=("hospitalization_id", "count"),
            aki_cases=("aki", "sum"),
        )
        .reset_index()
        .sort_values("week_start")
    )
    weekly["incidence_pct"] = 100.0 * weekly["aki_cases"] / weekly["admissions"]
    # 95% Wilson CI for a proportion
    from math import sqrt

    z = 1.96
    p = weekly["aki_cases"] / weekly["admissions"]
    n = weekly["admissions"]
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = (z * (p * (1 - p) / n + z**2 / (4 * n**2)).pow(0.5)) / denom
    weekly["ci_low_pct"] = 100 * (center - halfwidth)
    weekly["ci_high_pct"] = 100 * (center + halfwidth)

    print(f"Weeks with data: {len(weekly):,}")
    weekly.head()
    return (weekly,)


@app.cell
def _(weekly):
    # --- Summary stats ---
    print("Overall (pooled 2018-2024):")
    tot_adm = weekly["admissions"].sum()
    tot_aki = weekly["aki_cases"].sum()
    print(f"  admissions:  {tot_adm:,}")
    print(f"  aki cases:   {tot_aki:,}")
    print(f"  incidence:   {100 * tot_aki / tot_adm:.2f}%")
    print()
    print(f"Weekly incidence percentile summary:")
    print(weekly["incidence_pct"].describe(percentiles=[0.05, 0.5, 0.95]).round(2))
    return


@app.cell
def _():
    # --- Shared plotting import (hoisted so both plot cells can use `alt`) ---
    import altair as alt

    return (alt,)


@app.cell
def _(alt, mo, weekly):
    # --- Plot weekly incidence ---
    line = (
        alt.Chart(weekly)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("week_start:T", title="Week"),
            y=alt.Y("incidence_pct:Q", title="AKI incidence (% of admissions)"),
            tooltip=[
                alt.Tooltip("week_start:T", title="Week"),
                alt.Tooltip("admissions:Q", title="Admissions"),
                alt.Tooltip("aki_cases:Q", title="AKI cases"),
                alt.Tooltip("incidence_pct:Q", format=".2f", title="Incidence (%)"),
            ],
        )
    )
    band = (
        alt.Chart(weekly)
        .mark_area(opacity=0.2, color="#1f77b4")
        .encode(
            x="week_start:T",
            y="ci_low_pct:Q",
            y2="ci_high_pct:Q",
        )
    )
    chart = mo.ui.altair_chart(
        (band + line).properties(
            width=900,
            height=360,
            title="Weekly AKI incidence, adult admissions, UCMC 2018-2024",
        )
    )
    chart
    return


@app.cell
def _(cohort_aki, pd):
    # --- Weekly incidence stratified by age decile (10-year bands starting at 18) ---
    df_age = cohort_aki.copy()
    df_age["admission_local"] = df_age["admission_dttm"].dt.tz_convert("US/Central")
    df_age["week_start"] = (
        df_age["admission_local"]
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
        .dt.start_time
    )

    age_bins = [18, 28, 38, 48, 58, 68, 78, 88, 200]
    age_labels = ["18-27", "28-37", "38-47", "48-57", "58-67", "68-77", "78-87", "88+"]
    df_age["age_band"] = pd.cut(
        df_age["age_at_admission"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )
    df_age = df_age.dropna(subset=["age_band"])

    weekly_age = (
        df_age.groupby(["week_start", "age_band"], observed=True)
        .agg(
            admissions=("hospitalization_id", "count"),
            aki_cases=("aki", "sum"),
        )
        .reset_index()
        .sort_values(["age_band", "week_start"])
    )
    weekly_age["incidence_pct"] = (
        100.0 * weekly_age["aki_cases"] / weekly_age["admissions"]
    )

    # Wilson 95% CI
    z_a = 1.96
    p_a = weekly_age["aki_cases"] / weekly_age["admissions"]
    n_a = weekly_age["admissions"]
    denom_a = 1 + z_a**2 / n_a
    center_a = (p_a + z_a**2 / (2 * n_a)) / denom_a
    halfw_a = (z_a * (p_a * (1 - p_a) / n_a + z_a**2 / (4 * n_a**2)).pow(0.5)) / denom_a
    weekly_age["ci_low_pct"] = 100 * (center_a - halfw_a)
    weekly_age["ci_high_pct"] = 100 * (center_a + halfw_a)

    print(f"Weekly x age-band rows: {len(weekly_age):,}")
    print("Pooled incidence by age band:")
    pooled = (
        df_age.groupby("age_band", observed=True)
        .agg(admissions=("hospitalization_id", "count"), aki_cases=("aki", "sum"))
        .assign(incidence_pct=lambda x: 100 * x["aki_cases"] / x["admissions"])
    )
    print(pooled.round(2))
    return (weekly_age,)


@app.cell
def _(alt, mo, weekly_age):
    # --- Plot weekly incidence stratified by age band ---
    line_age = (
        alt.Chart(weekly_age)
        .mark_line()
        .encode(
            x=alt.X("week_start:T", title="Week"),
            y=alt.Y("incidence_pct:Q", title="AKI incidence (% of admissions)"),
            color=alt.Color(
                "age_band:N",
                title="Age band",
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("week_start:T", title="Week"),
                alt.Tooltip("age_band:N", title="Age band"),
                alt.Tooltip("admissions:Q", title="Admissions"),
                alt.Tooltip("aki_cases:Q", title="AKI cases"),
                alt.Tooltip("incidence_pct:Q", format=".2f", title="Incidence (%)"),
            ],
        )
    )
    chart_age = mo.ui.altair_chart(
        line_age.properties(
            width=900,
            height=400,
            title="Weekly AKI incidence by age band, adult admissions, UCMC 2018-2024",
        )
    )
    chart_age
    return


@app.cell
def _(RESULTS_DIR, mo, weekly, weekly_age):
    # --- Write a clean multi-page PDF report to results/ ---
    from pathlib import Path
    from datetime import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = results_dir / "aki_weekly_incidence_report.pdf"

    # Pooled numbers for the summary
    tot_adm_pdf = int(weekly["admissions"].sum())
    tot_aki_pdf = int(weekly["aki_cases"].sum())
    overall_pct = 100 * tot_aki_pdf / tot_adm_pdf if tot_adm_pdf else 0.0

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: title + summary ---
        fig0, ax0 = plt.subplots(figsize=(8.5, 11))
        ax0.axis("off")
        ax0.text(
            0.05,
            0.92,
            "Weekly Incidence of AKI in Adults\nHospitalized at UCMC (2018–2024)",
            fontsize=20,
            fontweight="bold",
            va="top",
        )
        ax0.text(
            0.05,
            0.80,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n"
            f"Data source: CLIF v2.1 (UCMC)\n"
            f"AKI definition: KDIGO Serum Creatinine (0.3 mg/dL / 48h OR 1.5× baseline / 7d)\n"
            f"Cohort: age ≥ 18, admission_dttm in 2018-01-01…2024-12-31",
            fontsize=11,
            va="top",
            family="monospace",
        )
        ax0.text(
            0.05,
            0.60,
            "Pooled results\n"
            "--------------\n"
            f"Adult admissions:   {tot_adm_pdf:,}\n"
            f"AKI cases:          {tot_aki_pdf:,}\n"
            f"Overall incidence:  {overall_pct:.2f}%",
            fontsize=12,
            va="top",
            family="monospace",
        )
        ax0.text(
            0.05,
            0.35,
            "Figures\n"
            "-------\n"
            "1. Weekly AKI incidence (overall) with 95% Wilson CI band\n"
            "2. Weekly AKI incidence stratified by 10-year age band (18-27 … 88+)",
            fontsize=11,
            va="top",
            family="monospace",
        )
        pdf.savefig(fig0)
        plt.close(fig0)

        # --- Page 2: overall weekly incidence ---
        fig1, ax1 = plt.subplots(figsize=(11, 6.5))
        ax1.fill_between(
            weekly["week_start"],
            weekly["ci_low_pct"],
            weekly["ci_high_pct"],
            color="#1f77b4",
            alpha=0.2,
            label="95% Wilson CI",
        )
        ax1.plot(
            weekly["week_start"],
            weekly["incidence_pct"],
            color="#1f77b4",
            linewidth=1.2,
            label="Weekly incidence",
        )
        ax1.set_title(
            "Weekly AKI incidence, adult admissions, UCMC 2018-2024",
            fontsize=13,
            fontweight="bold",
        )
        ax1.set_xlabel("Week")
        ax1.set_ylabel("AKI incidence (% of admissions)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", frameon=False)
        fig1.autofmt_xdate()
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # --- Page 3: stratified by age band ---
        fig2, ax2 = plt.subplots(figsize=(11, 6.5))
        age_band_values = (
            list(weekly_age["age_band"].cat.categories)
            if hasattr(weekly_age["age_band"], "cat")
            else sorted(weekly_age["age_band"].unique())
        )
        cmap = plt.get_cmap("viridis")
        for i, ab in enumerate(age_band_values):
            sub = weekly_age[weekly_age["age_band"] == ab].sort_values("week_start")
            if sub.empty:
                continue
            ax2.plot(
                sub["week_start"],
                sub["incidence_pct"],
                color=cmap(i / max(1, len(age_band_values) - 1)),
                linewidth=1.1,
                label=str(ab),
            )
        ax2.set_title(
            "Weekly AKI incidence by age band, adult admissions, UCMC 2018-2024",
            fontsize=13,
            fontweight="bold",
        )
        ax2.set_xlabel("Week")
        ax2.set_ylabel("AKI incidence (% of admissions)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Age band", loc="upper right", frameon=False, ncol=2)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"PDF written to: {pdf_path}")
    mo.md(f"[Open the PDF report](file://{pdf_path})")
    return


@app.cell
def _(mo, weekly):
    mo.ui.table(weekly, page_size=20)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Notes and caveats

    - Baseline SCr uses the minimum value **within** the hospitalization. If outpatient
      creatinine is available and desired, extend baseline to include prior-30-day values.
    - Hospitalizations with **no** in-stay creatinine cannot be adjudicated and are
      counted as non-AKI in the denominator. Consider a sensitivity analysis restricted
      to `n_scr > 0`.
    - Encounters were **not stitched** (readmissions within 6h treated as separate). To
      stitch, use `clifpy.utils.stitching_encounters.stitch_encounters` and aggregate at
      `encounter_block` instead of `hospitalization_id`.
    - Urine-output KDIGO criterion is not applied; it requires reliable hourly UOP which
      is typically ICU-only.
    - ESRD/dialysis-dependent patients are **not excluded** in this first pass. To
      exclude: filter `hospital_diagnosis` for ICD codes `N185`, `N186`, `Z992` with
      `present_on_admission = True`, or exclude patients on chronic intermittent HD.
    """)
    return


if __name__ == "__main__":
    app.run()
