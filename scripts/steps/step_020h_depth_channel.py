#!/usr/bin/env python3
"""step_020h: field-depth ambient channel -- tested and rejected.

The empirical ambient map (step_020g) measures u_eff ~ rho^{0.51}
declining with |Z|.  This step audits the candidate that pair
screening tracks the local temporal-field DEPTH u(z) = int|u_z|.

Verdict: REJECTED as a derivation.  Two failure modes:

1.  Boundary artifact.  u_depth integrated to a fixed outer edge
    z_max is dominated by the remaining path length, not by local
    density: the fitted exponent sweeps 1.30 -> 0.11 as z_max runs
    400 -> 2000 pc, and the apparent rho^0.52 match at z_max = 600 pc
    is the numerical coincidence (z_max - z) ~ rho^{~0.5} over the
    sampled range.

2.  Wrong sign locally.  Depth accumulated over a physically
    motivated local window (W = 50-300 pc, the pair's screening
    column scale) INCREASES with |Z| (exponent -0.25 .. -0.02),
    because the underlying |u_z| rises with z.  Only windows large
    enough to touch the artificial boundary flip the sign.

Also recorded: the field-value amplitude of the disk depth is
~5e-9 in well-solve units (u_z is a gradient normalised to g_t;
the amplitude carries g_t/c^2), far too shallow to act as a
starvation ambient, and the whole-disk stochastic rms channel
(~0.12, nearly flat in z) has the wrong slope.

Exact nonlinear field census (field_census section).  In the
k-branch the flux J = |du|.du obeys a LINEAR Gauss law,
div J = rho e^{-u} ~ rho in the inter-well medium, so for any
source set J(x) is the Newtonian-like vector sum and the true
field gradient is |du| = sqrt(2 J / k) -- this IS the exact
nonlinear superposition (p-Laplacian reduction).  Applied to the
disk well network plus smooth components: the coherent vertical
field rises 0.032 -> 0.091 over z = 22 -> 347 pc (wrong sign),
the discreteness fluctuation is ~0.004 (40x short), and the
total |du| at a pair location rises with z.  The Galactic
in-plane field |du_R| ~ sqrt(2 g_r/(k g_t)) ~ 0.23 is the only
quantity with the measured amplitude but is z-flat.  Inter-well
saddle amplitudes are ~0 (u_c - A ln(D/2r*) < 0 for D ~ 2 pc)
and the accumulated amplitude ~0.14 is density-independent by
construction.  CONCLUSION: no real-space field quantity supplies
u_eff ~ 0.12-0.20 declining with z; u_eff is an effective
in-medium parameter, not a boundary ambient.

Surviving statement: the measured law u_eff ~ rho^0.40 (020g,
mass-kernel corrected) stands; m_eff^2 ~ rho (verified in the
model) and the k-branch planar u_z ~ J^1/2 scaling identify the
density-coupled mechanism class; the operative projection is the
in-medium truncation of the pair halo by the well network -- the
AUD-3 constraint-slice derivation.

Inputs:  two-branch planar solve (step_020d conventions),
         step_020g ambient-map output.
Outputs: results/outputs/020h_depth_channel.json
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step_020d_disk_ambient as s20d

OUTDIR = os.path.join(HERE, "..", "..", "results", "outputs")
G020G = os.path.join(OUTDIR, "020g_ambient_map.json")

Z_STRATA = np.array([22, 68, 121, 196, 347])   # median |Z| pc, step_020g


def depth_to_boundary(z, uz, zmax):
    zz = z[z <= zmax]
    uzz = np.abs(uz[z <= zmax])
    if zmax > zz[-1]:
        ext = np.linspace(zz[-1], zmax, 500)
        zz = np.concatenate([zz, ext])
        uzz = np.concatenate([uzz, np.full_like(ext, uzz[-1])])
    udep = np.zeros_like(zz)
    for i in range(zz.size - 2, -1, -1):
        udep[i] = udep[i + 1] + 0.5 * (uzz[i] + uzz[i + 1]) * (zz[i + 1] - zz[i])
    return zz, udep


def local_depth(z, uz, zz0, W):
    m = (z >= zz0) & (z <= zz0 + W)
    return float(np.trapezoid(np.abs(uz[m]), z[m]))


def run():
    z, Jz, uz, ut = s20d.vertical_solve()
    rho = np.array([s20d.rho_disk(zz) for zz in Z_STRATA])

    boundary_scan = []
    for zmax in [400, 500, 600, 800, 1000, 1500, 2000]:
        zz, udep = depth_to_boundary(z, uz, zmax)
        ud = np.array([udep[int(np.argmin(np.abs(zz - v)))] for v in Z_STRATA])
        boundary_scan.append({
            "z_max_pc": zmax,
            "contrast_mid_over_halo": float(ud[0] / ud[-1]),
            "exponent_vs_rho": float(np.polyfit(np.log(rho), np.log(ud), 1)[0])})

    window_scan = []
    for W in [50, 100, 200, 300, 500]:
        ud = np.array([local_depth(z, uz, v, W) for v in Z_STRATA])
        window_scan.append({
            "W_pc": W,
            "contrast_mid_over_halo": float(ud[0] / ud[-1]),
            "exponent_vs_rho": float(np.polyfit(np.log(rho), np.log(ud), 1)[0])})

    # field-value amplitude of the depth (well-solve units)
    C2 = 8.99e16  # c^2 m^2/s^2
    amp_pc = s20d.GT / C2 * s20d.PC   # amplitude per (u_z x pc)
    _, udep600 = depth_to_boundary(z, uz, 600)
    depth_amp = float(udep600[int(np.argmin(np.abs(z)))] * amp_pc)

    # --- exact nonlinear field census ---------------------------
    # k-branch Gauss law: div F = rho (dimensionless source), with
    # F = (k/sqrt2) |du| du the flux vector -> F(x) is exactly the
    # Newtonian-like field of the source set (charges q_i = GM_i/g_t),
    # and the field gradient is |du| = sqrt(sqrt2 |F| / k).
    K = 16.03
    qF_sun = s20d.G * s20d.MSUN / s20d.GT          # m^2 per Msun
    # coherent vertical flux of the disk column, F_z = 2 pi G Sigma/g_t
    zz2 = np.linspace(0, 2000, 2001)

    rho2 = np.array([s20d.rho_disk(v) for v in zz2])

    def Sigma_encl(z0):                            # Msun/pc^2
        m = zz2 <= z0
        return 2 * np.trapezoid(rho2[m], zz2[m]) \
            if z0 > 0 else 0.0

    kg_m2 = s20d.MSUN / s20d.PC**2                 # Msun/pc^2 -> kg/m^2
    F_z = np.array([2 * np.pi * s20d.G * Sigma_encl(v) * kg_m2
                    / s20d.GT for v in Z_STRATA])
    F_R = 1.9e-10 / s20d.GT                        # Galactic in-plane
    # discreteness fluctuation in F, nearest-well dominated:
    # delta F^2 ~ 4 pi n <q^2> / r_min
    Mbar = math.exp(math.log(0.35) + 0.5 * 0.55**2)
    Mm2 = math.exp(2 * math.log(0.35) + 2 * 0.55**2)
    nstar = (0.040 * np.exp(-Z_STRATA / 300.)
             + 0.005 * np.exp(-Z_STRATA / 900.)) / Mbar     # /pc^3
    r_min = (3. / (4 * np.pi * nstar))**(1. / 3.) * s20d.PC  # m
    dF2 = 4 * np.pi * (nstar / s20d.PC**3) * (qF_sun**2 * Mm2) / r_min
    # field gradient from each component and the vector sum
    u_of_F = lambda F: np.sqrt(np.sqrt(2) * F / K)
    u_coh_z = u_of_F(F_z)
    u_fluc = u_of_F(np.sqrt(dF2))
    u_R = u_of_F(F_R)
    u_tot = u_of_F(np.sqrt(F_R**2 + F_z**2 + dF2))
    field_census = {
        "z_med_pc": Z_STRATA.tolist(),
        "u_vertical_component": [float(v) for v in u_coh_z],
        "u_discreteness_fluc": [float(v) for v in u_fluc],
        "u_galactic_inplane": float(u_R),
        "u_total": [float(v) for v in u_tot],
        "F_z": [float(v) for v in F_z],
        "F_R": float(F_R),
        "note": "exact k-branch superposition: F obeys a linear "
                "Gauss law, |du| = sqrt(sqrt2|F|/k).  In-plane flux "
                "dominates (F_R~0.62) and is z-flat; F_z rises with "
                "z; discreteness ~50x short.  Total |du| ~0.23 flat "
                "-- right amplitude, wrong slope."
    }

    g = json.load(open(G020G))
    ueff = [s["u_eff_from_alpha"] for s in g["free_fit_strata"]]

    out = {
        "verdict": "depth channel REJECTED as the ambient derivation",
        "boundary_scan": boundary_scan,
        "local_window_scan": window_scan,
        "disk_depth_field_amplitude_midplane": depth_amp,
        "field_census": field_census,
        "measured_u_eff_alpha_channel": ueff,
        "measured_exponent_alpha_channel":
            g["empirical_exponent_alpha_channel"],
        "failure_modes": [
            "exponent is an artifact of the outer boundary: "
            "1.30 -> 0.11 over z_max = 400 -> 2000 pc",
            "local-window depth (screening-column scale W<=300pc) "
            "has the wrong sign",
            "field-value depth ~5e-9 amplitude: cannot act as a "
            "starvation ambient"],
        "surviving": [
            "measured law u_eff ~ rho^0.40 (step_020g, "
            "per-stratum mass kernels)",
            "m_eff^2 ~ rho density coupling (model-exact)",
            "k-branch planar u_z ~ J^1/2 structure",
            "exact field census: every real-space |du| channel "
            "excluded (wrong sign, 40x short, or z-flat)",
            "operative projection: in-medium truncation of the pair "
            "halo by the well network -- AUD-3 constraint slice"]
    }
    print("z_max scan:", [(b["z_max_pc"], round(b["exponent_vs_rho"], 2))
                          for b in boundary_scan])
    print("W scan:", [(w["W_pc"], round(w["exponent_vs_rho"], 2))
                      for w in window_scan])
    print("depth amplitude (well units): %.2e" % depth_amp)
    with open(os.path.join(OUTDIR, "020h_depth_channel.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(OUTDIR, "020h_depth_channel.json"))


if __name__ == "__main__":
    run()
