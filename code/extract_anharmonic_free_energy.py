#!/usr/bin/env python3
"""F(V) global quadratic EOS fit + thermal expansion for bcc Mo and bcc Zr.

For every material in MATERIALS, walk
    <root>/vol_<RATIO>/<prefix>_<TEMP>K/tdep_calculations/convergence_data.txt
collect the free energy at every (volume ratio, nominal temperature),
fit F(V) = a*V^2 + b*V + c to ALL volume points at that temperature,
compute thermal expansion (per-interval and full-range),
print two summary tables, save them as CSV, and produce one final figure.
"""

# =====================================================================
# CONFIGURATION  (edit me)
# =====================================================================
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

MATERIALS = {
    "Mo": BASE_DIR / "outputs_mo_eqvol",
    "Zr": BASE_DIR / "outputs_zr_small_eqvol",
}

# Discard convergence_data rows whose Free_Energy is below this value
# (used to ignore the spurious -39 eV/atom Zr rows).
THRESHOLD = -20.0

# Per-material volume ratios to exclude from the analysis entirely.
# Each excluded ratio is matched with a small float tolerance against vol_*.
EXCLUDE_VOLS = {
    "Mo": [],
    "Zr": [],
}
EXCLUDE_VOL_TOL = 1e-4

# Outputs.
OUT_DIR             = SCRIPT_DIR
EOS_TABLE_CSV       = OUT_DIR / "eos_global_quadratic_results.csv"
EXPANSION_TABLE_CSV = OUT_DIR / "thermal_expansion_intervals_global_quadratic.csv"
FIGURE_PATH         = OUT_DIR / "free_energy_global_quadratic_Mo_Zr.png"

# Quality-control threshold for the "large_RMSE" note.
RMSE_WARN_THRESHOLD_MEV = 50.0


# =====================================================================
# Imports
# =====================================================================
import csv
import math
import re
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

NAN = float("nan")


# =====================================================================
# Data collection
# =====================================================================

_TEMP_NAME_RE = re.compile(r"_(\d+(?:\.\d+)?)K$")
_VOL_RE       = re.compile(r"vol_([0-9.]+)")


def _parse_convergence_data(path: Path, threshold: float):
    """Latest data row whose Free_Energy_eV_per_atom is >= threshold,
    or None if nothing qualifies."""
    valid = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            step = int(parts[0]); free_e = float(parts[2])
        except ValueError:
            continue
        if free_e >= threshold:
            valid.append((step, free_e))
    if not valid:
        return None
    return max(valid, key=lambda sf: sf[0])


def collect_free_energies(root: Path, threshold: float):
    """Return list of (vol_ratio, T_nominal_K, F_eV_per_atom)."""
    rows = []
    for vol_dir in sorted(root.glob("vol_*")):
        if not vol_dir.is_dir():
            continue
        m = _VOL_RE.search(vol_dir.name)
        if not m:
            continue
        vol_ratio = float(m.group(1))
        for temp_dir in sorted(vol_dir.iterdir()):
            if not temp_dir.is_dir():
                continue
            tm = _TEMP_NAME_RE.search(temp_dir.name)
            if not tm:
                continue
            t_nominal = float(tm.group(1))
            cf = temp_dir / "tdep_calculations" / "convergence_data.txt"
            if not cf.is_file():
                continue
            r = _parse_convergence_data(cf, threshold)
            if r is None:
                continue
            _, free_e = r
            rows.append((vol_ratio, t_nominal, free_e))
    return rows


# =====================================================================
# Global quadratic fit
# =====================================================================

def fit_global_quadratic(volumes, energies):
    """F(V) = a V^2 + b V + c on all (V, F) points.
    Returns dict: a, b, c, Veq, Feq, rmse_meV, status, notes."""
    volumes = np.asarray(volumes, float)
    energies = np.asarray(energies, float)
    if len(volumes) < 3:
        return dict(a=NAN, b=NAN, c=NAN, Veq=NAN, Feq=NAN, rmse_meV=NAN,
                    status="fit_failed", notes=["too_few_points_(<3)"])
    try:
        a, b, c = np.polyfit(volumes, energies, 2)
    except Exception as ex:
        return dict(a=NAN, b=NAN, c=NAN, Veq=NAN, Feq=NAN, rmse_meV=NAN,
                    status="fit_failed", notes=[f"polyfit_error: {ex}"])
    pred  = a * volumes ** 2 + b * volumes + c
    resid = pred - energies
    rmse  = float(np.sqrt(np.mean(resid * resid)) * 1000.0)  # eV -> meV
    if a <= 0:
        return dict(a=float(a), b=float(b), c=float(c),
                    Veq=NAN, Feq=NAN, rmse_meV=rmse,
                    status="non-convex", notes=["a<=0_(non_convex)"])
    Veq = -b / (2.0 * a)
    Feq = a * Veq * Veq + b * Veq + c
    notes = []
    if Veq < volumes.min() or Veq > volumes.max():
        notes.append("Veq_outside_sampled_range")
    if rmse > RMSE_WARN_THRESHOLD_MEV:
        notes.append(f"large_RMSE_(>{RMSE_WARN_THRESHOLD_MEV:.0f}_meV)")
    return dict(a=float(a), b=float(b), c=float(c),
                Veq=float(Veq), Feq=float(Feq), rmse_meV=rmse,
                status="ok", notes=notes)


# =====================================================================
# Thermal expansion
# =====================================================================

def thermal_expansion_intervals(temps_veqs):
    """Adjacent (T1->T2) alpha/beta/percent from list of (T, Veq)."""
    pairs = sorted(temps_veqs, key=lambda tv: tv[0])
    out = []
    for i in range(len(pairs) - 1):
        T1, V1 = pairs[i]
        T2, V2 = pairs[i + 1]
        if (np.isfinite(V1) and np.isfinite(V2)
                and V1 > 0 and V2 > 0 and T2 != T1):
            beta  = (math.log(V2) - math.log(V1)) / (T2 - T1)
            alpha = beta / 3.0
            pct   = 100.0 * (V2 / V1 - 1.0)
            notes = ""
        else:
            beta = alpha = pct = NAN
            notes = "interval_invalid"
        out.append(dict(T1_K=T1, T2_K=T2, Veq_T1=V1, Veq_T2=V2,
                        **{"beta_K^-1": beta, "alpha_K^-1": alpha},
                        percent_volume_change=pct, notes=notes))
    return out


def thermal_expansion_full_range(temps_veqs):
    """Use lowest-T and highest-T valid Veq."""
    valid = [(T, V) for (T, V) in temps_veqs
             if np.isfinite(V) and V > 0]
    if len(valid) < 2:
        return NAN, NAN, NAN
    valid.sort(key=lambda tv: tv[0])
    T1, V1 = valid[0]
    T2, V2 = valid[-1]
    if T1 == T2:
        return NAN, NAN, NAN
    beta  = (math.log(V2) - math.log(V1)) / (T2 - T1)
    alpha = beta / 3.0
    pct   = 100.0 * (V2 / V1 - 1.0)
    return alpha, beta, pct


# =====================================================================
# Output: tables, CSV
# =====================================================================

EOS_COLS = [
    "material", "T_K",
    "Veq", "Feq_eV_per_atom",
    "a", "b", "c",
    "RMSE_meV_per_atom",
    "fit_status", "notes",
    "alpha_full_range_K^-1", "beta_full_range_K^-1",
    "percent_volume_change_full_range",
]
EXP_COLS = [
    "material",
    "T1_K", "T2_K", "Veq_T1", "Veq_T2",
    "beta_K^-1", "alpha_K^-1", "percent_volume_change",
    "notes",
]


def _fmt(v, w):
    if v is None:
        return "".rjust(w)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN".rjust(w)
        if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.4e}".rjust(w)
        return f"{v:.6g}".rjust(w)
    if isinstance(v, int):
        return f"{v:d}".rjust(w)
    s = str(v)
    if len(s) > w:
        s = s[:w-1] + "…"
    return s.rjust(w)


def _print_table(rows, cols, widths, title):
    print(f"\n=== {title} ===")
    header = "  ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r[c], widths[c]) for c in cols))


def _save_csv(rows, cols, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# =====================================================================
# Figure
# =====================================================================

_TITLE_MAP = {"Mo": "bcc Mo", "Zr": "bcc Zr"}


def make_figure(data, fits, out_path):
    materials = list(MATERIALS.keys())
    n = len(materials)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)
    axes = axes[0]

    for ax, mat in zip(axes, materials):
        per_T = data.get(mat, {})
        if not per_T:
            ax.text(0.5, 0.5, f"No data for {mat}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(_TITLE_MAP.get(mat, mat))
            continue

        temps_sorted = sorted(per_T)
        cmap = plt.cm.viridis(np.linspace(0.0, 0.95,
                                          max(len(temps_sorted), 2)))
        all_vols = [v for T in per_T for v, _ in per_T[T]]
        all_Fs   = [F for T in per_T for _, F in per_T[T]]
        v_lo, v_hi = min(all_vols), max(all_vols)
        Vrange = (v_lo - 0.005, v_hi + 0.005)
        F_lo, F_hi = min(all_Fs), max(all_Fs)
        span   = F_hi - F_lo if F_hi > F_lo else 1.0
        Yrange = (F_lo - 0.10 * span, F_hi + 0.10 * span)
        Vc = np.linspace(Vrange[0], Vrange[1], 200)

        for color, T in zip(cmap, temps_sorted):
            pts = sorted(per_T[T])
            volumes = np.array([v for v, _ in pts])
            energies = np.array([F for _, F in pts])
            ax.scatter(volumes, energies, color=color, marker="o", s=42,
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=f"{int(T)} K")

            fit = fits.get((mat, T))
            if fit is None or fit["status"] != "ok":
                continue  # skip non-convex / failed
            a, b, c = fit["a"], fit["b"], fit["c"]
            ax.plot(Vc, a * Vc * Vc + b * Vc + c,
                    color=color, linewidth=1.4, zorder=2, alpha=0.85)
            if np.isfinite(fit["Veq"]):
                ax.scatter([fit["Veq"]], [fit["Feq"]],
                           color=color, marker="*", s=160,
                           edgecolor="black", linewidth=0.6, zorder=4)

        ax.set_xlabel(r"Volume ratio  $V/V_0$", fontsize=12)
        ax.set_ylabel(r"Free energy  $F$  (eV/atom)", fontsize=12)
        ax.set_title(_TITLE_MAP.get(mat, mat), fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(Vrange)
        ax.set_ylim(Yrange)
        ax.legend(title="Temperature", loc="best",
                  fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Interpretation
# =====================================================================

def print_interpretation(data, fits, full_range_per_mat):
    print("\n=== Interpretation (global quadratic) ===")
    for mat in MATERIALS:
        if mat not in data or not data[mat]:
            print(f"\n{mat}:  (no data)")
            continue

        per_T = data[mat]
        print(f"\n{mat}:")
        veq_seq = []
        all_notes = []
        any_non_ok = False
        for T in sorted(per_T):
            fit = fits.get((mat, T))
            if fit is None:
                continue
            v = fit["Veq"]
            v_str = f"{v:.6f}" if np.isfinite(v) else "NaN"
            line = (f"  T={int(T)} K  Veq={v_str}"
                    f"  status={fit['status']}"
                    f"  RMSE={fit['rmse_meV']:.3f} meV/atom")
            if fit["notes"]:
                line += f"  notes={'; '.join(fit['notes'])}"
                all_notes.extend(fit["notes"])
            print(line)
            if np.isfinite(v):
                veq_seq.append(v)
            if fit["status"] != "ok":
                any_non_ok = True
                all_notes.append(fit["status"])

        alpha, beta, pct = full_range_per_mat[mat]
        if np.isfinite(alpha):
            sign = "POSITIVE" if alpha > 0 else ("NEGATIVE" if alpha < 0 else "ZERO")
        else:
            sign = "INDETERMINATE"
        print(f"  full-range alpha = {alpha:.3e} K^-1   ({sign})")
        print(f"  full-range beta  = {beta:.3e} K^-1")
        print(f"  full-range %ΔV   = {pct:.4f} %")

        if len(veq_seq) >= 2:
            mono_up = all(veq_seq[i+1] >= veq_seq[i] for i in range(len(veq_seq)-1))
            mono_dn = all(veq_seq[i+1] <= veq_seq[i] for i in range(len(veq_seq)-1))
            monotonic = mono_up or mono_dn
        else:
            monotonic = True
        print(f"  Veq(T) monotonic: {monotonic}")
        if all_notes:
            uniq = sorted(set(all_notes))
            print(f"  Warnings: {', '.join(uniq)}")

        # Material-specific physical-plausibility wording
        if mat == "Mo":
            if np.isfinite(alpha) and 4e-6 <= alpha <= 7e-6:
                print(f"  -> alpha = {alpha:.2e} K^-1 is physically "
                      f"reasonable in magnitude for bcc Mo.")
            elif np.isfinite(alpha) and alpha > 0:
                print(f"  -> alpha = {alpha:.2e} K^-1 is positive, but "
                      f"outside the typical 5-6e-6 K^-1 range for bcc Mo.")
        elif mat == "Zr":
            if np.isfinite(alpha) and 5e-6 <= alpha <= 5e-5:
                msg = (f"  -> alpha = {alpha:.2e} K^-1 is physically "
                       f"plausible for bcc Zr")
                if any_non_ok or not monotonic or any(
                        n.startswith("large_RMSE")
                        or n.startswith("Veq_outside") for n in all_notes):
                    msg += " — but PROVISIONAL given the warnings above."
                else:
                    msg += "."
                print(msg)
            elif np.isfinite(alpha) and alpha > 0:
                print(f"  -> alpha = {alpha:.2e} K^-1 is positive but "
                      f"outside the ~1e-5 K^-1 range expected for bcc Zr.")


# =====================================================================
# Main
# =====================================================================

def main():
    # Step 1: collect raw F(V) per (material, temperature)
    data = {}
    for mat, root in MATERIALS.items():
        root = Path(root)
        if not root.is_dir():
            continue
        rows = collect_free_energies(root, THRESHOLD)
        excluded = EXCLUDE_VOLS.get(mat, [])
        per_T = {}
        for vol_ratio, T, F in rows:
            if any(abs(vol_ratio - bad) < EXCLUDE_VOL_TOL for bad in excluded):
                continue
            per_T.setdefault(T, []).append((vol_ratio, F))
        for T in per_T:
            per_T[T].sort()
        data[mat] = per_T

    # Step 2: global quadratic fit per (material, T)
    fits = {}
    veq_per_mat = {m: [] for m in data}
    for mat, per_T in data.items():
        for T in sorted(per_T):
            volumes = [v for v, _ in per_T[T]]
            energies = [F for _, F in per_T[T]]
            fit = fit_global_quadratic(volumes, energies)
            fits[(mat, T)] = fit
            if np.isfinite(fit["Veq"]):
                veq_per_mat[mat].append((T, fit["Veq"]))

    # Step 3: full-range thermal expansion per material
    full_range_per_mat = {
        mat: thermal_expansion_full_range(veq_per_mat[mat])
        for mat in data
    }

    # Step 4: build EOS-table rows (one per material, T)
    eos_rows = []
    for mat in data:
        alpha_full, beta_full, pct_full = full_range_per_mat[mat]
        for T in sorted(data[mat]):
            fit = fits[(mat, T)]
            eos_rows.append({
                "material": mat,
                "T_K": T,
                "Veq": fit["Veq"],
                "Feq_eV_per_atom": fit["Feq"],
                "a": fit["a"], "b": fit["b"], "c": fit["c"],
                "RMSE_meV_per_atom": fit["rmse_meV"],
                "fit_status": fit["status"],
                "notes": "; ".join(fit["notes"]),
                "alpha_full_range_K^-1": alpha_full,
                "beta_full_range_K^-1":  beta_full,
                "percent_volume_change_full_range": pct_full,
            })

    # Step 5: build adjacent-interval rows (one per material, interval)
    expansion_rows = []
    for mat in data:
        for iv in thermal_expansion_intervals(veq_per_mat[mat]):
            expansion_rows.append({"material": mat, **iv})

    # Step 6: print tables
    eos_widths = {
        "material": 8, "T_K": 6,
        "Veq": 10, "Feq_eV_per_atom": 14,
        "a": 12, "b": 12, "c": 14,
        "RMSE_meV_per_atom": 12,
        "fit_status": 12, "notes": 32,
        "alpha_full_range_K^-1": 12, "beta_full_range_K^-1": 12,
        "percent_volume_change_full_range": 10,
    }
    exp_widths = {
        "material": 8, "T1_K": 6, "T2_K": 6,
        "Veq_T1": 10, "Veq_T2": 10,
        "beta_K^-1": 12, "alpha_K^-1": 12,
        "percent_volume_change": 12,
        "notes": 16,
    }
    _print_table(eos_rows,       EOS_COLS, eos_widths,
                 "EOS Global Quadratic Results (per material, temperature)")
    _print_table(expansion_rows, EXP_COLS, exp_widths,
                 "Thermal Expansion (adjacent intervals)")

    # Step 7: write CSVs
    _save_csv(eos_rows,       EOS_COLS, EOS_TABLE_CSV)
    _save_csv(expansion_rows, EXP_COLS, EXPANSION_TABLE_CSV)
    print(f"\nSaved CSV: {EOS_TABLE_CSV}")
    print(f"Saved CSV: {EXPANSION_TABLE_CSV}")

    # Step 8: figure
    make_figure(data, fits, FIGURE_PATH)
    print(f"Saved figure: {FIGURE_PATH}")

    # Step 9: interpretation
    print_interpretation(data, fits, full_range_per_mat)


if __name__ == "__main__":
    main()
