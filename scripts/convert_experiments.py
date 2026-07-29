#!/usr/bin/env -S uv run --quiet --with openpyxl --with pandas python3
"""
Convert the published rebound datasets into tidy CSVs under data/experiments/.

TWO SOURCES, and the distinction matters because only one carries the target variable.

  A. "LowWeberDropRebound - Data.xlsx" -- published figure data for Alventosa et al. 2023
     (references/BouncingDroplets.tex). Each sheet holds side-by-side blocks for
     Experiment / DNS / KM model, and the EXPERIMENT blocks carry UNCERTAINTIES.
       Figure 5(b) -> CONTACT TIME tc/t_sigma vs We          <-- the target variable
       Figure 5(a) -> coefficient of restitution vs We
       Figure 6(b) -> contact radius rc/R as a time series
     This is the file to compare against.

  B. "DataForReboundsOnly.xlsx" (km-dropplet-onto-bath/matlab/1_code/Figures/) -- 627 raw
     rebound measurements, single fluid. Much larger, but it has NO CONTACT TIME: its columns
     are sigma, mu, rho, Dn, class, R, Vi, Vo, ho, eps, We, Bo, Oh. Retained as a broad
     secondary CoR dataset only.

NOT INTERPRETED, deliberately, because guessing would give a plausible-looking wrong
comparison:
  * B's `cor` is published as a restitution coefficient but is NOT Vo/Vi (the ratio spans
    -87..+34 across the rows), so it is computed some other way. Passed through verbatim with
    the raw Vi/Vo retained so any definition can be recomputed downstream.
  * B's `ho` column: metres, 0.26..5.73 times R, only weakly correlated with We (r = 0.47),
    meaning undocumented, and the sister repo's own experimental max-deflection scatter is
    commented out. Carried through as `ho_m` WITHOUT being called a deflection.

Run: ./scripts/convert_experiments.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "experiments"
FIGDATA = Path.home() / "Downloads" / "LowWeberDropRebound - Data.xlsx"
REBOUNDS = (Path.home() /
            "Documents/Github/km-dropplet-onto-bath/matlab/1_code/Figures/DataForReboundsOnly.xlsx")

# (sheet, output name, value column, {source: (first_col, has_uncertainty)})
FIG_SPECS = [
    ("Figure 5(b)", "contact_time_vs_we.csv", "tc_over_tsigma",
     {"experiment": (1, True), "dns": (8, False), "km_model": (13, False)}),
    ("Figure 5(a)", "cor_vs_we.csv", "cor",
     {"experiment": (1, True), "dns": (8, False), "km_model": (13, False)}),
]


def convert_figure(sheet, outname, value, blocks):
    d = pd.read_excel(FIGDATA, sheet_name=sheet, header=None)
    frames = []
    for src, (c0, unc) in blocks.items():
        # experiment: We, We_unc, value, value_unc, Oh, Bo ; others: We, value, Oh, Bo
        cols = [c0 + i for i in (range(6) if unc else range(4))]
        names = (["We", "We_unc", value, f"{value}_unc", "Oh", "Bo"] if unc
                 else ["We", value, "Oh", "Bo"])
        blk = d.iloc[2:, cols].copy()
        blk.columns = names
        for c in names:
            blk[c] = pd.to_numeric(blk[c], errors="coerce")
        blk = blk.dropna(subset=["We", value]).reset_index(drop=True)
        if not unc:
            blk["We_unc"] = pd.NA
            blk[f"{value}_unc"] = pd.NA
        blk["source"] = src
        frames.append(blk[["source", "We", "We_unc", value, f"{value}_unc", "Oh", "Bo"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUTDIR / outname, index=False)
    print(f"  {outname:30s} {len(out):5d} rows  {out.source.value_counts().to_dict()}")
    e = out[out.source == "experiment"]
    # guard: some blocks contain exact zeros for the value, so mask before dividing
    nz = e[e[value] != 0]
    rel = (nz[f"{value}_unc"] / nz[value]).median() if len(nz) else float("nan")
    print(f"  {'':30s} experiment: We {e.We.min():.4g}..{e.We.max():.4g}, "
          f"{value} {e[value].min():.4g}..{e[value].max():.4g}, median rel unc {rel:.1%}")


def convert_contact_radius():
    """Figure 6(b): rc/R time series. Two experiment blocks at different We, plus DNS and KM."""
    d = pd.read_excel(FIGDATA, sheet_name="Figure 6(b)", header=None)
    blocks = {"experiment_lowWe": 1, "dns": 7, "km_model": 13, "experiment_highWe": 19}
    frames = []
    for series, c0 in blocks.items():
        blk = d.iloc[3:, [c0 + i for i in range(5)]].copy()
        blk.columns = ["t_over_tsigma", "rc_over_R", "We", "Oh", "Bo"]
        for c in blk.columns:
            blk[c] = pd.to_numeric(blk[c], errors="coerce")
        blk = blk.dropna(subset=["t_over_tsigma", "rc_over_R"]).reset_index(drop=True)
        blk["source"] = "experiment" if series.startswith("experiment") else series
        blk["series"] = series
        frames.append(blk[["source", "series", "t_over_tsigma", "rc_over_R", "We", "Oh", "Bo"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUTDIR / "contact_radius_timeseries.csv", index=False)
    print(f"  {'contact_radius_timeseries.csv':30s} {len(out):5d} rows  "
          f"{out.series.value_counts().to_dict()}")


RAW_COLS = {1: "sigma_N_per_m", 2: "mu_Pa_s", 3: "rho_kg_per_m3", 4: "Dn_mm", 6: "outcome",
            7: "R_m", 8: "R_std_m", 9: "Vi_m_per_s", 10: "Vi_std_m_per_s",
            11: "Vo_m_per_s", 12: "Vo_std_m_per_s", 13: "ho_m",
            15: "cor", 16: "We", 17: "Bo", 18: "Oh"}


def convert_rebounds():
    raw = pd.read_excel(REBOUNDS, sheet_name="Sheet1", header=None)
    df = raw.iloc[3:][list(RAW_COLS)].rename(columns=RAW_COLS)
    for c in df.columns:
        if c != "outcome":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["We", "Bo", "Oh", "cor"]).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    df["cor_from_velocities"] = df["Vo_m_per_s"] / df["Vi_m_per_s"]
    df.to_csv(OUTDIR / "rebound_low_weber_raw.csv", index=False)
    print(f"  {'rebound_low_weber_raw.csv':30s} {len(df):5d} rows  "
          f"We {df.We.min():.4g}..{df.We.max():.4g}, Oh {df.Oh.min():.4g}..{df.Oh.max():.4g}"
          f"   (NO contact time in this file)")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in (FIGDATA, REBOUNDS) if not p.exists()]
    if missing:
        sys.exit("missing source(s):\n  " + "\n  ".join(str(m) for m in missing))
    print("A. published figure data (Alventosa et al. 2023) -- carries the TARGET variable:")
    for spec in FIG_SPECS:
        convert_figure(*spec)
    convert_contact_radius()
    print("\nB. raw rebound measurements -- broad CoR only:")
    convert_rebounds()


if __name__ == "__main__":
    main()
