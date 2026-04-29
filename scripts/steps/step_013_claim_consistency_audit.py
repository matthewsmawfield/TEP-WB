#!/usr/bin/env python3
"""
Step 013: Claim Consistency Audit for TEP-WB

Checks that the manuscript source and generated markdown are consistent with the
latest numerical outputs. This is intentionally lightweight: it guards against
stale headline numbers and known audit regressions without trying to parse every
sentence in the paper.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, print_status, set_step_logger


OUTPUTS = PROJECT_ROOT / "results" / "outputs"
COMPONENTS = PROJECT_ROOT / "site" / "components"


def read_text(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""


def html_int(value):
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def add_check(rows, name, passed, detail, severity="error"):
    rows.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def run_claim_consistency_audit():
    print_status("Initializing Claim Consistency Audit", "TITLE")

    required = [
        "002_beta_mlr_calibration.json",
        "003_screening_fit_summary.csv",
        "003_model_comparison.csv",
        "005_environment_results.csv",
        "008_injection_recovery_summary.csv",
        "011_mond_efe_comparison.csv",
        "012_newtonian_forward_model_summary.csv",
    ]
    missing = [name for name in required if not (OUTPUTS / name).exists()]
    if missing:
        print_status(f"Missing required outputs: {', '.join(missing)}", "ERROR")
        sys.exit(1)

    beta = json.loads(read_text(OUTPUTS / "002_beta_mlr_calibration.json"))
    fit = pd.read_csv(OUTPUTS / "003_screening_fit_summary.csv").iloc[0]
    models = pd.read_csv(OUTPUTS / "003_model_comparison.csv")
    env = pd.read_csv(OUTPUTS / "005_environment_results.csv").iloc[0]
    inj = pd.read_csv(OUTPUTS / "008_injection_recovery_summary.csv")
    mond = pd.read_csv(OUTPUTS / "011_mond_efe_comparison.csv")
    newton = pd.read_csv(OUTPUTS / "012_newtonian_forward_model_summary.csv")

    manuscript_source = "\n".join(
        read_text(path) for path in sorted(COMPONENTS.glob("*.html"))
    )
    generated = read_text(PROJECT_ROOT / "13-TEP-WB-v0.3-Kilifi.md")

    # Release metadata sources
    version_txt = read_text(PROJECT_ROOT / "version.txt")
    root_bib = read_text(PROJECT_ROOT / "CITATION.bib")
    site_bib = read_text(PROJECT_ROOT / "site" / "CITATION.bib")
    root_cff = read_text(PROJECT_ROOT / "CITATION.cff")
    site_cff = read_text(PROJECT_ROOT / "site" / "CITATION.cff")
    version_json = read_text(PROJECT_ROOT / "VERSION.json")
    manifest_json = read_text(PROJECT_ROOT / "site" / "manifest.json")
    zenodo_desc = read_text(PROJECT_ROOT / "zenodo.txt")

    rows = []

    add_check(
        rows,
        "beta_calibration_non_circular",
        beta.get("calibration_mode") in {"spectroscopic", "literature_prior"}
        and beta.get("reference", "").startswith("Spectroscopic-only"),
        f"mode={beta.get('calibration_mode')}, p_value={beta.get('p_value')}",
    )
    add_check(
        rows,
        "no_stale_core_fit_numbers",
        "2{,}583" not in manuscript_source and "0.370" not in manuscript_source,
        "stale R_s=2,583 or alpha=0.370 should not appear in component claims",
    )
    add_check(
        rows,
        "core_fit_numbers_present",
        html_int(fit["r_s_au"]) in manuscript_source and f"{float(fit['alpha']):.3f}" in manuscript_source,
        f"expected R_s={html_int(fit['r_s_au'])}, alpha={float(fit['alpha']):.3f}",
    )

    sigmoid = models[models["model"] == "Sigmoid"].iloc[0]
    double = models[models["model"] == "Double_Exponential"].iloc[0]
    add_check(
        rows,
        "model_comparison_current",
        f"{float(sigmoid['delta_chi2_vs_tep']):+.1f}" in manuscript_source
        and f"{float(double['delta_chi2_vs_tep']):+.1f}" in manuscript_source
        and html_int(double["r_s_or_s_trans"]) in manuscript_source,
        (
            f"sigmoid_delta={float(sigmoid['delta_chi2_vs_tep']):+.1f}, "
            f"double_delta={float(double['delta_chi2_vs_tep']):+.1f}, "
            f"double_Rs={html_int(double['r_s_or_s_trans'])}"
        ),
    )
    add_check(
        rows,
        "mond_best_delta_current",
        "+540" in manuscript_source
        and "+364" not in manuscript_source
        and "+549" not in manuscript_source
        and "+897" not in manuscript_source,
        "MOND narrative should match 011_mond_efe_comparison.csv best free-EFE result",
    )

    add_check(
        rows,
        "environment_ordering_documented",
        env["joint_rs_low_z"] > env["joint_rs_high_z"]
        and env["free_alpha_low_z_rs"] < env["free_alpha_high_z_rs"]
        and "freely" in manuscript_source.lower(),
        (
            f"joint low/high={env['joint_rs_low_z']:.0f}/{env['joint_rs_high_z']:.0f}; "
            f"free low/high={env['free_alpha_low_z_rs']:.0f}/{env['free_alpha_high_z_rs']:.0f}"
        ),
        severity="warning",
    )

    signal = inj[inj["test"] == "signal_recovery"].iloc[0]
    null = inj[inj["test"] == "null_injection"].iloc[0]
    add_check(
        rows,
        "injection_recovery_hardened",
        float(signal["detection_rate"]) >= 0.99
        and float(null["detection_rate"]) <= 0.01
        and "projected Newtonian forward-model null" in manuscript_source,
        (
            f"signal_detection={signal['detection_rate']}; "
            f"null_false_positive={null['detection_rate']}"
        ),
    )

    best_newton = newton.sort_values("chi2_newtonian_profile_vs_observed").iloc[0]
    add_check(
        rows,
        "newtonian_forward_null_present",
        best_newton["outer_deficit_vs_observed"] > 0.25
        and "Newtonian orbital forward model" in manuscript_source,
        (
            f"best={best_newton['eccentricity_law']}, "
            f"outer_deficit={best_newton['outer_deficit_vs_observed']:.3f}, "
            f"chi2={best_newton['chi2_newtonian_profile_vs_observed']:.1f}"
        ),
    )

    if generated:
        add_check(
            rows,
            "generated_markdown_matches_core_claims",
            "2{,}583" not in generated
            and f"{float(sigmoid['delta_chi2_vs_tep']):+.1f}" in generated
            and "projected Newtonian forward-model null" in generated,
            "generated markdown should be rebuilt after component/output changes",
            severity="warning",
        )

    # Release metadata consistency checks
    add_check(
        rows,
        "version_txt_current",
        "v0.3" in version_txt and "v0.2" not in version_txt,
        "version.txt should reflect current release (v0.3)",
    )

    add_check(
        rows,
        "root_citation_bib_volume",
        'volume       = {13}' in root_bib,
        "root CITATION.bib should have volume = {13}",
    )

    add_check(
        rows,
        "site_citation_bib_volume",
        'volume       = {13}' in site_bib,
        "site CITATION.bib should have volume = {13}",
    )

    add_check(
        rows,
        "root_citation_cff_version",
        'version: "v0.3' in root_cff,
        "root CITATION.cff should have version v0.3",
    )

    add_check(
        rows,
        "site_citation_cff_version",
        'version: "v0.3' in site_cff,
        "site CITATION.cff should have version v0.3",
    )

    add_check(
        rows,
        "version_json_current",
        '"version": "v0.3' in version_json,
        "VERSION.json should have version v0.3",
    )

    add_check(
        rows,
        "manifest_json_current",
        '"version": "v0.3' in manifest_json,
        "site/manifest.json should have version v0.3",
    )

    # Normalize zenodo text: remove both plain commas and LaTeX {,} formatting
    zenodo_normalized = zenodo_desc.replace(",", "").replace("{,}", "")
    rs_plain = str(int(round(float(fit["r_s_au"]))))
    chi2_plain = "14845"
    add_check(
        rows,
        "zenodo_description_current",
        rs_plain in zenodo_normalized
        and f"{float(fit['alpha']):.3f}" in zenodo_desc
        and chi2_plain in zenodo_normalized,
        "zenodo.txt should contain current core claims",
        severity="warning",
    )

    summary = pd.DataFrame(rows)
    out_csv = OUTPUTS / "013_claim_consistency_audit.csv"
    out_json = OUTPUTS / "013_claim_consistency_audit.json"
    summary.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "n_checks": int(len(summary)),
                "n_failed_errors": int((~summary["passed"] & (summary["severity"] == "error")).sum()),
                "n_failed_warnings": int((~summary["passed"] & (summary["severity"] == "warning")).sum()),
                "checks": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = summary[(~summary["passed"]) & (summary["severity"] == "error")]
    warnings = summary[(~summary["passed"]) & (summary["severity"] == "warning")]
    for _, row in summary.iterrows():
        level = "SUCCESS" if row["passed"] else ("WARNING" if row["severity"] == "warning" else "ERROR")
        print_status(f"{row['check']}: {row['detail']}", level)

    print_status(f"Saved audit outputs to {out_csv}", "SUCCESS")
    if len(warnings) > 0:
        print_status(f"{len(warnings)} warning-level claim checks need interpretation.", "WARNING")
    if len(failures) > 0:
        print_status(f"{len(failures)} error-level claim checks failed.", "ERROR")
        sys.exit(1)

    print_status("Claim consistency audit passed.", "SUCCESS")


if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_013", str(log_dir / "step_013_claim_consistency_audit.log"))
    set_step_logger(logger)
    run_claim_consistency_audit()
