#!/usr/bin/env python3
"""Auxiliary run: two-centre pair profile at the DERIVED solar-circle
ambient u0 = 0.1629 (two-branch self-consistent scalar shear,
step_020).  Produces y_pair(d/r*) for the refined estimator.
"""
import importlib.util
import json
import os

import numpy as np

U0 = 0.16291036984356919
D_LIST = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


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
        # linear references (uniform ambient; coupling state clean)
        ref = {}
        for d in D_LIST:
            rho_AB = s59.src_rho((-d / 2.0, +d / 2.0))
            rho_B = s59.src_rho((+d / 2.0,))
            psi = s59.solve_linear(rho_AB, 0.0)
            psiB = s59.solve_linear(rho_B, 0.0)
            r_c = max(d / 3.0, 4.0 * s59.SIG)
            ref[d] = (s59.stress_z_linear(psi, 0.0, d / 2.0, r_c)
                      - s59.stress_z_linear(psiB, 0.0, d / 2.0, r_c))
            psi_d = s59.solve_linear(rho_AB, 0.0)
            ref_d = s59.direct_force(psi_d, rho_B, 0.0)
            ref[d] = (ref[d], ref_d)
        rows = []
        for d in D_LIST:
            rho_AB = s59.src_rho((-d / 2.0, +d / 2.0))
            rho_B = s59.src_rho((+d / 2.0,))
            r_c = max(d / 3.0, 4.0 * s59.SIG)
            psi_AB = s59.solve_with(rho_AB, U0, "two_branch")
            F_AB = s59.stress_z(psi_AB, U0, d / 2.0, r_c, "two_branch")
            psi_B = s59.solve_with(rho_B, U0, "two_branch")
            F_B = s59.stress_z(psi_B, U0, d / 2.0, r_c, "two_branch")
            F = F_AB - F_B
            Fd = s59.pair_force_direct(d, U0, "two_branch")
            rows.append({
                "d_over_rstar": d,
                "F_pair": F, "F_linear": ref[d][0],
                "F_pair_direct": Fd, "F_linear_direct": ref[d][1],
                "y_pair": F / ref[d][0],
                "y_pair_direct": Fd / ref[d][1]})
            print("d=%.2f y=%.4f y_dir=%.4f"
                  % (d, rows[-1]["y_pair"],
                     rows[-1]["y_pair_direct"]), flush=True)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)

    out = {"u0": U0, "sector": "two_branch", "rows": rows}
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "results", "outputs",
                        "020b_pair_profile_derived_u0.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
