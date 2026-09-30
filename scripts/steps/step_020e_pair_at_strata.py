#!/usr/bin/env python3
"""Auxiliary run: two-centre pair profile at the stratum ambients
inferred under the corrected estimator — u0 = 0.10 (halo) and
u0 = 0.23 (midplane).  Tests whether the pair suppression deepening
at higher ambient shrinks the required environmental contrast.
"""
import importlib.util
import json
import os

U0S = [0.10, 0.23]
D_LIST = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]


def load(rel):
    here = os.path.dirname(os.path.abspath(__file__))
    tep = os.path.join(here, "..", "..", "..", "TEP")
    spec = importlib.util.spec_from_file_location(
        "s59", os.path.join(tep, "scripts", "steps", rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    s59 = load("step_59_two_center.py")
    m = s59.m
    saved = {k: getattr(m, k) for k in
             ("STARVE", "STARVE_LAM", "KPHI_C", "KPHI_PERT",
              "PSI_FROZEN", "PHI0")}
    try:
        ref = {}
        for d in D_LIST:
            rho_AB = s59.src_rho((-d / 2.0, +d / 2.0))
            rho_B = s59.src_rho((+d / 2.0,))
            r_c = max(d / 3.0, 4.0 * s59.SIG)
            psi = s59.solve_linear(rho_AB, 0.0)
            psiB = s59.solve_linear(rho_B, 0.0)
            ref_s = (s59.stress_z_linear(psi, 0.0, d / 2.0, r_c)
                     - s59.stress_z_linear(psiB, 0.0, d / 2.0, r_c))
            ref_d = s59.direct_force(s59.solve_linear(rho_AB, 0.0),
                                     rho_B, 0.0)
            ref[d] = (ref_s, ref_d)
        allrows = {}
        for u0 in U0S:
            rows = []
            for d in D_LIST:
                rho_AB = s59.src_rho((-d / 2.0, +d / 2.0))
                rho_B = s59.src_rho((+d / 2.0,))
                r_c = max(d / 3.0, 4.0 * s59.SIG)
                psi_AB = s59.solve_with(rho_AB, u0, "two_branch")
                F_AB = s59.stress_z(psi_AB, u0, d / 2.0, r_c,
                                    "two_branch")
                psi_B = s59.solve_with(rho_B, u0, "two_branch")
                F_B = s59.stress_z(psi_B, u0, d / 2.0, r_c,
                                   "two_branch")
                F = F_AB - F_B
                Fd = s59.pair_force_direct(d, u0, "two_branch")
                rows.append({
                    "d_over_rstar": d,
                    "y_pair": F / ref[d][0],
                    "y_pair_direct": Fd / ref[d][1]})
                print("u0=%.2f d=%.2f y=%.4f y_dir=%.4f"
                      % (u0, d, rows[-1]["y_pair"],
                         rows[-1]["y_pair_direct"]), flush=True)
            allrows[str(u0)] = rows
    finally:
        for k, v in saved.items():
            setattr(m, k, v)

    out = {"sector": "two_branch", "u0s": U0S, "rows": allrows}
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "results", "outputs",
                        "020e_pair_profile_strata.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
