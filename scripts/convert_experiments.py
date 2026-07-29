#!/usr/bin/env -S uv run --quiet --with openpyxl --with pandas python3
"""
Convert the low-Weber rebound experiments from the sister repo's spreadsheet into a clean CSV.

SOURCE
  km-dropplet-onto-bath/matlab/1_code/Figures/DataForReboundsOnly.xlsx, Sheet1.
  Header on spreadsheet row 3, data from row 4. Every row is classified 'b' (bounce) -- the
  file is rebounds only, as its name says, so there are no float/merge outcomes to filter.

COLUMN MAPPING, taken from how lowWeberComparison.m:100-104 reads the same file (1-based
columns there): eps = 16, We = 17, Bo = 18, Oh = 19. Verified against the header text.

TWO THINGS DELIBERATELY NOT INTERPRETED, because they are not recoverable from the sheet and
guessing would produce a plausible-looking wrong comparison:

  eps  Published as the coefficient of restitution and used as such by lowWeberComparison.m
       (plotted on an $\\alpha$ axis). It is NOT simply Vo/Vi: over these 627 rows the ratio
       eps/(Vo/Vi) ranges from -87 to +34, so eps is computed by some other route (plausibly an
       energy ratio referenced to the z=0 crossing, per Galeano-Rios et al.). Passed through
       verbatim, alongside the raw Vi/Vo so any definition can be recomputed downstream.
  ho   Units of metres, 0.26 to 5.73 times R, correlating only weakly with We (r = 0.47). Its
       meaning is undocumented and lowWeberComparison.m never uses it -- its experimental
       max-deflection scatter is commented out. Carried through as `ho_m` WITHOUT being
       labelled a deflection.

Run:  ./scripts/convert_experiments.py     (or: uv run --with openpyxl --with pandas scripts/convert_experiments.py)
"""
import sys
from pathlib import Path
import pandas as pd

SRC = Path.home() / "Documents/Github/km-dropplet-onto-bath/matlab/1_code/Figures/DataForReboundsOnly.xlsx"
OUT = Path(__file__).resolve().parent.parent / "data/experiments/rebound_low_weber.csv"

COLS = {1: "sigma_N_per_m", 2: "mu_Pa_s", 3: "rho_kg_per_m3", 4: "Dn_mm", 6: "outcome",
        7: "R_m", 8: "R_std_m", 9: "Vi_m_per_s", 10: "Vi_std_m_per_s",
        11: "Vo_m_per_s", 12: "Vo_std_m_per_s", 13: "ho_m",
        15: "cor", 16: "We", 17: "Bo", 18: "Oh"}

def main():
    if not SRC.exists():
        sys.exit(f"source spreadsheet not found: {SRC}")
    raw = pd.read_excel(SRC, sheet_name="Sheet1", header=None)
    df = raw.iloc[3:][list(COLS)].rename(columns=COLS)
    for c in df.columns:
        if c != "outcome":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["We", "Bo", "Oh", "cor"]).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    # derived, clearly marked as ours rather than published
    df["cor_from_velocities"] = df["Vo_m_per_s"] / df["Vi_m_per_s"]
    df = df.round({"We": 8, "Bo": 8, "Oh": 8, "cor": 8, "cor_from_velocities": 8})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"wrote {OUT.relative_to(OUT.parent.parent.parent)}  ({len(df)} rows, dropped {before-len(df)} blank)")
    print(f"  outcomes      : {df.outcome.value_counts().to_dict()}")
    print(f"  fluids        : {df.groupby(['sigma_N_per_m','mu_Pa_s','rho_kg_per_m3']).ngroups} distinct")
    for c in ("We", "Bo", "Oh", "cor"):
        print(f"  {c:4s}          : {df[c].min():.5g} .. {df[c].max():.5g}   median {df[c].median():.5g}")
    print(f"  cor vs Vo/Vi  : ratio {(df.cor/df.cor_from_velocities).min():.1f} .. "
          f"{(df.cor/df.cor_from_velocities).max():.1f}  -- NOT the same quantity, see docstring")

if __name__ == "__main__":
    main()
