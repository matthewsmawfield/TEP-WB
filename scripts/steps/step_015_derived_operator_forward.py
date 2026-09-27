"""
Step 015: Derived-operator forward-model test (issue 0-27).

Folds the master-action screening operator S_eff(r) = S_env * [1+(R_s/r)^k]^-1
through the observable pipeline and compares the binned median profile with the
canonical 19-bin measurement:

  1. each real binary's projected separation s_p is deprojected to a true
     3D separation r via a random sky orientation (r = s_p / sin(theta));
  2. the per-binary shell radius follows the derived law R_s = sqrt(GM/g_t)
     (Smawfield 2025a, sec.7), with lognormal environmental scatter;
  3. the asymptotic pairwise response S_env (Galactic ambient floor) carries
     lognormal scatter across environments;
  4. unresolved-triple contamination uses the corrected vector-addition
     kinematics of step_009/010 (positive-definite |v_orb + v_phot|);
  5. multiplicative Newtonian scatter models the eccentricity/orientation
     width of the per-binary v_tan/v_c distribution;
  6. binned medians are normalised to the inner baseline exactly as in the
     measurement.

Outputs the chi2_red of predicted vs observed profiles across the operator
exponent k and the environmental scale multiplier, plus the exact
flux-conserving profile y(s) of the resolved nested operator
(R = S_env^2 * y(s) for comparable-mass pairs; Paper 0 step_30). The
resolved operator's transition-region slope (~4/3) is tested directly
rather than approximated by a power law.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from utils.logging_utils import print_status
except ImportError:
    def print_status(msg, level="INFO"):
        print(f"[{level}] {msg}")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
OBS_PATH = PROJECT_ROOT / "results" / "outputs" / "003_screening_test_results.csv"
OUT_DIR = PROJECT_ROOT / "results" / "outputs"

G_MSUN = 1.327e20       # m^3 s^-2
G_T = 3.4e-10           # canonical transition acceleration, m s^-2
AU_M = 1.496e11         # m per AU
BINS = np.logspace(np.log10(50), np.log10(30000), 20)
NMC = 8
SEED = 271828


def y_profile(x):
    """Exact flux-conserving shear profile: y(1 + y^2 x^-4) = 1.
    Transition-region slope ~4/3; saturates to 1 outside the shell."""
    x = np.asarray(x)
    lo = np.zeros_like(x)
    hi = np.ones_like(x)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = mid * (1.0 + mid * mid / np.clip(x, 1e-12, None) ** 4) - 1.0
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


def forward_median(sp, M, Rs0, k, Senv, sig_env, sig_Rs, scale,
                   f_tr=0.10, vf_a=0.35, noise=0.20, nmc=NMC, rng=None):
    """Monte-Carlo forward model -> normalised binned medians.
    k = 'y' selects the exact flux-conserving profile (resolved operator);
    numeric k selects the [1+(Rs/r)^k]^-1 power-law family."""
    meds = np.zeros((nmc, BINS.size - 1))
    n = sp.size
    for m in range(nmc):
        mu = rng.uniform(-1.0, 1.0, n)
        r = sp / np.sqrt(np.clip(1.0 - mu**2, 1e-6, 1.0))
        Rs = Rs0 * scale * np.exp(rng.normal(0.0, sig_Rs, n))
        Se = np.clip(Senv * np.exp(rng.normal(0.0, sig_env, n)), 0.05, 0.95)
        if k == "y":
            S = Se * y_profile(r / Rs)
        else:
            S = Se / (1.0 + (Rs / r) ** k)
        v = np.sqrt(1.0 + 2.0 * S)
        # corrected vector-addition triple contamination (step_009/010)
        mask = rng.random(n) < f_tr
        vf = vf_a * np.sqrt(sp / 1000.0)
        cpsi = rng.uniform(-1.0, 1.0, n)
        vt = np.sqrt(np.clip(v**2 + vf**2 + 2.0 * v * vf * cpsi, 0.0, None))
        v = np.where(mask, vt, v)
        v *= np.exp(rng.normal(0.0, noise, n))
        for i in range(BINS.size - 1):
            mm = (sp >= BINS[i]) & (sp < BINS[i + 1])
            meds[m, i] = np.median(v[mm]) if mm.sum() > 30 else np.nan
    med = np.nanmedian(meds, axis=0)
    return med / np.nanmedian(med[:6])


def main():
    print_status("Step 015: derived-operator forward-model test", "HEADER")
    rng = np.random.default_rng(SEED)

    df = pd.read_parquet(DATA_PATH)
    df = df[(df["ruwe1"] < 1.2) & (df["ruwe2"] < 1.2)].dropna(
        subset=["sep_AU", "mass_total"])
    obs = pd.read_csv(OBS_PATH)
    v_obs = obs["v_tilde_norm"].values
    e_obs = obs["sem_norm"].values

    sp = df["sep_AU"].values
    M = df["mass_total"].values
    Rs0 = np.sqrt(G_MSUN * M / G_T) / AU_M
    print_status(f"{sp.size} binaries; median R_s = "
                 f"{np.median(Rs0):.0f} AU", "INFO")

    rows = []
    for k in (0.8, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, "y"):
        best = (np.inf, None)
        for scale in (0.4, 0.5, 0.65, 0.8, 1.0):
            for Senv in (0.35, 0.45, 0.55):
                for sig_Rs in (0.30, 0.45):
                    med = forward_median(sp, M, Rs0, k, Senv, 0.3,
                                         sig_Rs, scale, rng=rng)
                    c = np.nansum(((v_obs - med) / e_obs) ** 2) / 17.0
                    if c < best[0]:
                        best = (c, (scale, Senv, sig_Rs, med.copy()))
        scale, Senv, sig_Rs, med = best[1]
        rows.append({"k": k, "chi2_red": best[0], "scale": scale,
                     "S_env": Senv, "sigma_logRs": sig_Rs})
        print_status(f"k = {k}: best chi2_red = {best[0]:.1f} "
                     f"(scale {scale}, S_env {Senv}, sig_Rs {sig_Rs})",
                     "RESULT")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "015_derived_operator_forward.csv", index=False)

    # save best-profile comparison across all candidates
    profiles = pd.DataFrame({
        "sep_AU": obs["sep_AU"], "observed": v_obs, "sem": e_obs})
    for kk, sc, se, sR in [(r["k"], r["scale"], r["S_env"],
                           r["sigma_logRs"])
                          for _, r in res.iterrows()]:
        md = forward_median(sp, M, Rs0, kk, se, 0.3, sR, sc, nmc=6, rng=rng)
        tag = kk if kk == "y" else f"{kk:.1f}"
        profiles[f"pred_k{tag}"] = md
    profiles.to_csv(OUT_DIR / "015_derived_operator_profiles.csv",
                    index=False)

    print_status("Saved 015_derived_operator_forward.csv and "
                 "015_derived_operator_profiles.csv", "SUCCESS")


if __name__ == "__main__":
    main()
