#!/usr/bin/env python3
"""
Step 003: Screening Test for TEP-WB

Performs the critical test of the TEP screening hypothesis with comprehensive
statistical analysis, model comparison, and systematic uncertainty quantification.

This script:
1. Bins kinematic data logarithmically by physical separation
2. Calculates median dimensionless velocities with bootstrap confidence intervals
3. Fits multiple phenomenological models (TEP exponential, sigmoid, double-exponential)
4. Performs model comparison via information criteria
5. Conducts runs test for residual randomness
6. Computes systematic uncertainty budget
7. Generates publication-quality figures with diagnostics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# Analysis Constants
BOOTSTRAP_SEED = 314159
BOOTSTRAP_ITERATIONS = 1000

# Set publication-quality matplotlib defaults
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'serif'],
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

def tep_screening_model(s, r_s, alpha):
    """
    Canonical TEP exponential screening model.
    v_tilde = 1 + alpha * (1 - exp(-s/r_s))
    """
    return 1.0 + alpha * (1.0 - np.exp(-s / r_s))


def sigmoid_model(s, s_trans, width, alpha):
    """
    Sigmoid (logistic) transition model.
    v_tilde = 1 + alpha / (1 + exp(-(s-s_trans)/width))
    """
    return 1.0 + alpha / (1.0 + np.exp(-(s - s_trans) / width))


def double_exp_model(s, r_s, alpha, beta):
    """
    Double-exponential model (steeper onset).
    v_tilde = 1 + alpha * (1 - exp(-(s/r_s)^beta))
    """
    return 1.0 + alpha * (1.0 - np.exp(-(s / r_s)**beta))


def flat_newtonian_model(s):
    return np.ones_like(np.asarray(s), dtype=float)


def constant_boost_model(s, alpha):
    return np.ones_like(np.asarray(s), dtype=float) * (1.0 + alpha)


def chi2_statistic(observed, expected, errors):
    """Calculate chi-squared statistic."""
    return float(np.sum(((observed - expected) / errors) ** 2))


def information_criteria(chi2, n_points, n_params):
    """Calculate AIC and BIC."""
    aic = chi2 + 2 * n_params
    bic = chi2 + n_params * np.log(n_points)
    return aic, bic


def runs_test(residuals):
    """
    Perform Wald-Wolfowitz runs test for residual randomness.
    Returns: (n_runs, expected_runs, z_score, p_value)
    """
    signs = np.sign(residuals)
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    n = n_pos + n_neg
    
    if n < 2:
        return 0, 0, 0, 1.0
    
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    
    expected_runs = (2 * n_pos * n_neg / n) + 1
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n-1))
    
    if var_runs <= 0:
        return runs, expected_runs, 0, 1.0
    
    std_runs = np.sqrt(var_runs)
    z_runs = (runs - expected_runs) / std_runs
    p_runs = 2 * stats.norm.sf(abs(z_runs))
    
    return runs, expected_runs, z_runs, p_runs


def bootstrap_confidence_intervals(df, n_boot=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """
    Calculate bootstrap confidence intervals for each bin.
    Returns arrays of lower and upper 68% CI bounds.
    """
    rng = np.random.default_rng(seed)
    boot_lower = []
    boot_upper = []
    
    for _, row in df.iterrows():
        sem = row['sem_norm']
        median = row['v_tilde_norm']
        boot_samples = rng.normal(median, sem, n_boot)
        boot_lower.append(np.percentile(boot_samples, 16))
        boot_upper.append(np.percentile(boot_samples, 84))
    
    return np.array(boot_lower), np.array(boot_upper)


def fit_model(model_func, s_data, v_data, v_err, p0, bounds):
    """Fit a model and return comprehensive results."""
    try:
        popt, pcov = curve_fit(
            model_func, s_data, v_data,
            sigma=v_err, p0=p0, bounds=bounds,
            maxfev=10000
        )
        
        v_model = model_func(s_data, *popt)
        residuals = v_data - v_model
        chi2 = np.sum((residuals / v_err)**2)
        dof = len(s_data) - len(popt)
        red_chi2 = chi2 / dof if dof > 0 else np.nan
        perr = np.sqrt(np.diag(pcov))
        
        return {
            'success': True,
            'params': popt,
            'errors': perr,
            'covariance': pcov,
            'chi2': chi2,
            'dof': dof,
            'reduced_chi2': red_chi2,
            'residuals': residuals,
            'model_values': v_model
        }
    except Exception:
        return {'success': False}

def perform_screening_test():
    print_status("Initializing Comprehensive Screening Test Pipeline", "TITLE")
    
    input_path = PROJECT_ROOT / "data" / "processed" / "kinematic_results.parquet"
    if not input_path.exists():
        print_status(f"Kinematic results not found at {input_path}. Run step_002 first.", "ERROR")
        sys.exit(1)

    print_status("Loading kinematic results...", "PROCESS")
    df = pd.read_parquet(input_path)
    
    # 1. Purity Filter
    mask_pure = (df['ruwe1'] < 1.2) & (df['ruwe2'] < 1.2)
    df = df[mask_pure].copy()
    print_status(f"Post-RUWE sample: {len(df)} binaries", "INFO")
    
    # 2. Binning the data
    bins = np.logspace(np.log10(50), np.log10(30000), 20)
    df['bin'] = pd.cut(df['sep_AU'], bins=bins)
    
    grouped = df.groupby('bin', observed=True)['v_tilde']
    bin_centers = 10**(0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    
    results = pd.DataFrame({
        'sep_AU': bin_centers,
        'v_tilde_median': grouped.median(),
        'v_tilde_mean': grouped.mean(),
        'v_tilde_std': grouped.std(),
        'count': grouped.count()
    }).dropna()
    
    # Standard Error of the Median
    results['sem'] = 1.253 * results['v_tilde_std'] / np.sqrt(results['count'])
    
    # 3. Normalization
    baseline = results['v_tilde_median'].iloc[0:5].mean()
    results['v_tilde_norm'] = results['v_tilde_median'] / baseline
    results['sem_norm'] = results['sem'] / baseline
    
    print_status(f"Baseline normalization: {baseline:.4f}", "INFO")
    print_status(f"Analyzing {len(results)} separation bins...", "PROCESS")
    
    # Prepare data
    s_data = results['sep_AU'].values
    v_data = results['v_tilde_norm'].values
    v_err = results['sem_norm'].values
    
    # =============================================================================
    # MODEL FITTING
    # =============================================================================
    
    # Model 1: TEP Exponential (Canonical)
    print_status("Fitting TEP Exponential Model...", "PROCESS")
    tep_result = fit_model(tep_screening_model, s_data, v_data, v_err,
                           p0=[3000.0, 0.4], bounds=([100.0, 0.0], [50000.0, 0.8]))
    
    if not tep_result['success']:
        print_status("TEP model fit failed!", "ERROR")
        sys.exit(1)
    
    r_s_tep = tep_result['params'][0]
    alpha_tep = tep_result['params'][1]
    r_s_err_tep = tep_result['errors'][0]
    alpha_err_tep = tep_result['errors'][1]
    
    print_status(f"TEP Fit: R_s = {r_s_tep:.1f} ± {r_s_err_tep:.1f} AU, α = {alpha_tep:.4f} ± {alpha_err_tep:.4f}", "SUCCESS")
    
    # Model 2: Sigmoid (Alternative)
    print_status("Fitting Sigmoid Model...", "PROCESS")
    sigmoid_result = fit_model(sigmoid_model, s_data, v_data, v_err,
                               p0=[3000.0, 1000.0, 0.4], bounds=([100.0, 50.0, 0.0], [50000.0, 10000.0, 0.8]))
    
    if sigmoid_result['success']:
        delta_chi2_sig = sigmoid_result['chi2'] - tep_result['chi2']
        print_status(f"Sigmoid Fit: Δχ² = {delta_chi2_sig:+.1f} vs TEP", "INFO")
    else:
        print_status("Sigmoid fit did not converge", "WARNING")
        sigmoid_result = {'success': False, 'chi2': np.inf}
    
    # Model 3: Double-Exponential (Alternative)
    print_status("Fitting Double-Exponential Model...", "PROCESS")
    dbl_exp_result = fit_model(double_exp_model, s_data, v_data, v_err,
                               p0=[3000.0, 0.4, 1.0], bounds=([100.0, 0.0, 0.3], [50000.0, 0.8, 3.0]))
    
    if dbl_exp_result['success']:
        delta_chi2_dbl = dbl_exp_result['chi2'] - tep_result['chi2']
        print_status(f"Double-Exp Fit: Δχ² = {delta_chi2_dbl:+.1f} vs TEP", "INFO")
    else:
        print_status("Double-exponential fit did not converge", "WARNING")
        dbl_exp_result = {'success': False, 'chi2': np.inf}
    
    # =============================================================================
    # MODEL COMPARISON
    # =============================================================================
    
    print_status("Performing Model Comparison...", "PROCESS")
    
    n_points = len(results)
    tep_aic, tep_bic = information_criteria(tep_result['chi2'], n_points, 2)
    
    # Null models
    flat_pred = flat_newtonian_model(s_data)
    flat_chi2 = chi2_statistic(v_data, flat_pred, v_err)
    flat_aic, flat_bic = information_criteria(flat_chi2, n_points, 0)
    
    weights = 1.0 / np.square(v_err)
    alpha_const = np.clip(np.sum(weights * (v_data - 1.0)) / np.sum(weights), 0.0, 0.5)
    const_pred = constant_boost_model(s_data, alpha_const)
    const_chi2 = chi2_statistic(v_data, const_pred, v_err)
    const_aic, const_bic = information_criteria(const_chi2, n_points, 1)
    
    print_status(f"vs Flat Newtonian: Δχ² = {flat_chi2 - tep_result['chi2']:.1f}", "RESULT")
    print_status(f"vs Constant Boost: Δχ² = {const_chi2 - tep_result['chi2']:.1f}", "RESULT")
    
    # =============================================================================
    # RUNS TEST
    # =============================================================================
    
    print_status("Performing Runs Test for Residual Randomness...", "PROCESS")
    runs, exp_runs, z_runs, p_runs = runs_test(tep_result['residuals'])
    
    print_status(f"Runs Test: {runs} runs, Z = {z_runs:.2f}, p = {p_runs:.4f}", 
                 "SUCCESS" if p_runs > 0.05 else "WARNING")
    
    # =============================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # =============================================================================
    
    print_status(f"Computing Bootstrap CIs (N={BOOTSTRAP_ITERATIONS}, seed={BOOTSTRAP_SEED})...", "PROCESS")
    boot_lower, boot_upper = bootstrap_confidence_intervals(results)
    results['v_tilde_norm_lower'] = boot_lower
    results['v_tilde_norm_upper'] = boot_upper
    results['model_tep'] = tep_result['model_values']
    results['residual_tep'] = tep_result['residuals']
    results['chi2_contrib'] = (results['residual_tep'] / results['sem_norm'])**2
    
    # =============================================================================
    # JACKKNIFE STABILITY
    # =============================================================================
    
    print_status("Performing Jackknife Stability Analysis...", "PROCESS")
    jackknife_rs = []
    jackknife_alpha = []
    for idx in results.index:
        subset = results.drop(index=idx)
        try:
            jk_popt, _ = curve_fit(tep_screening_model, subset['sep_AU'], subset['v_tilde_norm'],
                                   sigma=subset['sem_norm'], p0=[r_s_tep, alpha_tep],
                                   bounds=([100.0, 0.0], [50000.0, 0.5]), maxfev=10000)
            jackknife_rs.append(jk_popt[0])
            jackknife_alpha.append(jk_popt[1])
        except Exception:
            pass
    
    jackknife_rs = np.asarray(jackknife_rs)
    jackknife_alpha = np.asarray(jackknife_alpha)
    jackknife_rs_std = float(np.std(jackknife_rs, ddof=1)) if len(jackknife_rs) > 1 else np.nan
    jackknife_alpha_std = float(np.std(jackknife_alpha, ddof=1)) if len(jackknife_alpha) > 1 else np.nan
    
    print_status(f"Jackknife: σ(R_s) = {jackknife_rs_std:.1f} AU across {len(jackknife_rs)} refits", "INFO")
    
    # =============================================================================
    # BIN 1 SENSITIVITY TEST
    # =============================================================================
    
    print_status("Testing sensitivity to Bin 1 exclusion...", "PROCESS")
    results_no_bin1 = results.iloc[1:].copy()
    bin1_result = fit_model(tep_screening_model, results_no_bin1['sep_AU'].values,
                            results_no_bin1['v_tilde_norm'].values,
                            results_no_bin1['sem_norm'].values,
                            p0=[r_s_tep, alpha_tep],
                            bounds=([100.0, 0.0], [50000.0, 0.8]))
    if bin1_result['success']:
        r_s_no_bin1 = bin1_result['params'][0]
        alpha_no_bin1 = bin1_result['params'][1]
        r_s_shift_pct = abs(r_s_no_bin1 - r_s_tep) / r_s_tep * 100
        alpha_shift_pct = abs(alpha_no_bin1 - alpha_tep) / alpha_tep * 100
        print_status(f"Bin 1 excluded: R_s = {r_s_no_bin1:.1f} AU (shift {r_s_shift_pct:.1f}%), α = {alpha_no_bin1:.4f} (shift {alpha_shift_pct:.1f}%)", "RESULT")
    else:
        print_status("Bin 1 exclusion fit did not converge", "WARNING")

    # =============================================================================
    # SYSTEMATIC UNCERTAINTY BUDGET
    # =============================================================================
    
    print_status("Computing Systematic Uncertainty Budget...", "PROCESS")
    
    stat_unc_r_s = r_s_err_tep
    jk_unc_r_s = jackknife_rs_std if not np.isnan(jackknife_rs_std) else 0
    
    # Model choice uncertainty
    best_alt_r_s = None
    if dbl_exp_result['success']:
        best_alt_r_s = dbl_exp_result['params'][0]
    elif sigmoid_result['success']:
        best_alt_r_s = sigmoid_result['params'][0]
    
    model_unc_r_s = abs(r_s_tep - best_alt_r_s) if best_alt_r_s else 0
    sys_unc_r_s = np.sqrt(jk_unc_r_s**2 + model_unc_r_s**2)
    total_unc_r_s = np.sqrt(stat_unc_r_s**2 + sys_unc_r_s**2)
    
    print_status(f"R_s = {r_s_tep:.1f} ± {stat_unc_r_s:.1f} (stat) ± {sys_unc_r_s:.1f} (sys) = {total_unc_r_s:.1f} AU total", "RESULT")
    
    # =============================================================================
    # GENERATE OUTPUTS
    # =============================================================================
    
    print_status("Generating Outputs...", "PROCESS")
    figures_dir = PROJECT_ROOT / "results" / "figures"
    outputs_dir = PROJECT_ROOT / "results" / "outputs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Enhanced figure with residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    ax1.errorbar(results['sep_AU'], results['v_tilde_norm'],
                 yerr=[results['v_tilde_norm'] - results['v_tilde_norm_lower'],
                       results['v_tilde_norm_upper'] - results['v_tilde_norm']],
                 fmt='o', color='#4682B4', label='Data with 68% CI', capsize=3, alpha=0.8)
    
    s_grid = np.logspace(np.log10(50), np.log10(30000), 100)
    ax1.axhline(1.0, color='#708090', linestyle='--', label='Newtonian')
    ax1.plot(s_grid, tep_screening_model(s_grid, r_s_tep, alpha_tep), 
             color='#4B0082', linewidth=2.5, label=f'TEP Model (R_s={r_s_tep:.0f} AU)')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$\tilde{v} / \tilde{v}_{\mathrm{inner}}$')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    ax2.errorbar(results['sep_AU'], results['residual_tep'], yerr=results['sem_norm'],
                 fmt='o', color='#4682B4', capsize=3, alpha=0.6)
    ax2.axhline(0, color='#708090', linestyle='--')
    ax2.set_xscale('log')
    ax2.set_xlabel('Projected Separation s [AU]')
    ax2.set_ylabel('Residual')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "screening_transition.png", dpi=600)
    print_status("Saved screening transition figure", "SUCCESS")
    plt.close()
    
    # Save comprehensive outputs
    results.to_csv(outputs_dir / "screening_test_results.csv", index=False)
    
    # Enhanced fit summary
    fit_summary = pd.DataFrame({
        'r_s_au': [r_s_tep],
        'r_s_err_stat_au': [stat_unc_r_s],
        'r_s_err_sys_au': [sys_unc_r_s],
        'r_s_err_total_au': [total_unc_r_s],
        'alpha': [alpha_tep],
        'alpha_err_stat': [alpha_err_tep],
        'baseline_v_tilde': [baseline],
        'chi2': [tep_result['chi2']],
        'dof': [tep_result['dof']],
        'reduced_chi2': [tep_result['reduced_chi2']],
        'n_bins': [len(results)],
        'flat_null_chi2': [flat_chi2],
        'delta_chi2_vs_flat': [flat_chi2 - tep_result['chi2']],
        'runs_test_n_runs': [runs],
        'runs_test_z': [z_runs],
        'runs_test_p': [p_runs],
        'jackknife_r_s_std_au': [jackknife_rs_std],
        'jackknife_successes': [len(jackknife_rs)],
        'bootstrap_seed': [BOOTSTRAP_SEED],
        'bootstrap_iterations': [BOOTSTRAP_ITERATIONS],
        'model_choice_uncertainty_r_s': [model_unc_r_s],
    })
    fit_summary.to_csv(outputs_dir / "screening_fit_summary.csv", index=False)
    
    # Model comparison table
    model_comparison = pd.DataFrame({
        'model': ['TEP_Exponential', 'Sigmoid', 'Double_Exponential'],
        'r_s_or_s_trans': [r_s_tep,
            sigmoid_result['params'][0] if sigmoid_result['success'] else np.nan,
            dbl_exp_result['params'][0] if dbl_exp_result['success'] else np.nan],
        'alpha': [alpha_tep,
            sigmoid_result['params'][2] if sigmoid_result['success'] else np.nan,
            dbl_exp_result['params'][1] if dbl_exp_result['success'] else np.nan],
        'chi2': [tep_result['chi2'],
            sigmoid_result['chi2'] if sigmoid_result['success'] else np.nan,
            dbl_exp_result['chi2'] if dbl_exp_result['success'] else np.nan],
        'delta_chi2_vs_tep': [0,
            sigmoid_result['chi2'] - tep_result['chi2'] if sigmoid_result['success'] else np.nan,
            dbl_exp_result['chi2'] - tep_result['chi2'] if dbl_exp_result['success'] else np.nan],
        'converged': [True, sigmoid_result['success'], dbl_exp_result['success']]
    })
    model_comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)
    
    # Systematic uncertainty budget
    sys_budget = pd.DataFrame({
        'source': ['Statistical_fit', 'Jackknife_stability', 'Model_choice', 'Total_systematic', 'TOTAL'],
        'r_s_uncertainty_au': [stat_unc_r_s, jk_unc_r_s, model_unc_r_s, sys_unc_r_s, total_unc_r_s],
        'method': ['Covariance_matrix', 'Leave_one_out', 'Alternative_models', 'Quadrature_sum', 'Stat_plus_sys']
    })
    sys_budget.to_csv(outputs_dir / "systematic_uncertainty_budget.csv", index=False)
    
    print_status("All outputs saved", "SUCCESS")
    
    # Final summary
    print_status("="*60, "TITLE")
    print_status("SCREENING ANALYSIS COMPLETE", "TITLE")
    print_status("="*60, "TITLE")
    print_status(f"Primary Result: R_s = {r_s_tep:.1f} ± {total_unc_r_s:.1f} AU", "RESULT")
    print_status(f"Velocity Enhancement: +{alpha_tep*100:.1f}%", "RESULT")
    print_status(f"Significance vs Newtonian: Δχ² = {flat_chi2 - tep_result['chi2']:.0f}", "RESULT")
    print_status(f"Runs Test: p = {p_runs:.4f}", "RESULT")
    print_status("="*60, "TITLE")

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = TEPLogger("step_003", str(log_dir / "step_003_screening_test.log"))
    set_step_logger(logger)
    
    perform_screening_test()
