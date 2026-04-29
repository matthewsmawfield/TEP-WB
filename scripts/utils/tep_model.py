#!/usr/bin/env python3
"""
TEP Universal Model Utilities (v0.8 Jakarta/Kos)

Shared functions for computing TEP quantities across all research papers:
COS (Paper 10), H0 (Paper 11), JWST (Paper 12), WB (Paper 13).

This module provides a unified point of truth for:
1. Universal Couplings (KAPPA_GAL, ALPHA_INT)
2. Chronological Enhancement (Gamma_t)
3. Screening Mechanisms (Temporal Topology)
4. Kinematic Profile Models (Wide Binaries)

Author: Matthew L. Smawfield
Date: April 2026
"""

import numpy as np
import pandas as pd
from scipy import integrate

# =============================================================================
# 1. UNIVERSAL COUPLINGS & CONSTANTS
# =============================================================================

# CANONICAL OBSERVABLE RESPONSE COEFFICIENT (Paper 11)
# Measured from Cepheid period-luminosity residuals.
# Units: Magnitudes [mag]
KAPPA_GAL = 9.6e5
KAPPA_GAL_UNCERTAINTY = 4.0e5

# STELLAR EVOLUTION INDEX
# M/L ~ t^n from stellar isochrones.
ALPHA_NUCLEAR = 0.7

# POTENTIAL PARAMETERS
LOG_MH_REF = 12.0
PHI_REF_0 = 1.6e-7    # Dimensionless Phi/c^2 for 10^12 Msun halo at z=0
Z_REF = 5.5

# SCREENING SCALES
RHO_CRIT_G_CM3 = 20.0  # Temporal Topology saturation scale (g/cm^3)

# PHYSICAL CONSTANTS
C_LIGHT_KM_S = 2.99792458e5
G_NEWTON_PC_MSUN = 4.30091e-3  # (pc/Msun) * (km/s)^2
G_AU = 887.1                   # (km/s)^2 * AU / M_sun

# =============================================================================
# 1b. PIPELINE SHARED CONSTANTS (Wide Binary Sector)
# =============================================================================

# Standard binning for separation profiles (20 log-spaced bins from 50 to 30,000 AU)
GLOBAL_BINS = np.logspace(np.log10(50), np.log10(30000), 20)

# Bootstrap settings
BOOTSTRAP_SEED = 314159

# Injection-recovery / mock-catalog seed (used by step_008, step_009, step_010)
INJECTION_SEED = 271828

# Main-sequence mass-luminosity exponent L ~ M^MLR_EXPONENT (Salaris & Cassisi 2005)
MLR_EXPONENT = 3.5

# Conservative color-residual mass-bias prior used when no independent
# spectroscopic metallicity cache is available for beta_MLR calibration.
MLR_BETA_PRIOR = 1.5

# Fixed alpha for environmental tests (breaks degeneracy with R_s)
GLOBAL_ALPHA = 0.4

# =============================================================================
# 2. CHRONOLOGICAL ENHANCEMENT (Gamma_t)
# =============================================================================

def get_phi_from_log_mh(log_Mh):
    """Compute dimensionless virial potential Phi/c^2 at z=0."""
    return 1.6e-7 * (10**log_Mh / 1e12)**(2/3)

def compute_gamma_t(log_Mh, z, kappa=KAPPA_GAL, n=ALPHA_NUCLEAR):
    """
    Compute TEP chronological enhancement factor (Potential-Linear Form).
    
    Gamma_t = exp[ K * (Phi - Phi_ref) * sqrt(1+z) ]
    where K = kappa * ln(10) / (2.5 * n)
    """
    phi = get_phi_from_log_mh(log_Mh)
    k_exp = (kappa * np.log(10)) / (2.5 * n)
    z_scaling = np.sqrt(1 + z)
    argument = k_exp * (phi - PHI_REF_0) * z_scaling
    return np.exp(argument)

# =============================================================================
# 3. KINEMATIC & SCREENING MODELS
# =============================================================================

def tep_screening_model(s, r_s, alpha):
    """Canonical TEP exponential screening model for velocities."""
    return 1.0 + alpha * (1.0 - np.exp(-s / r_s))


def tep_screening_model_fixed_alpha(s, r_s, alpha=GLOBAL_ALPHA):
    """TEP screening model with fixed alpha (for environmental tests)."""
    return 1.0 + alpha * (1.0 - np.exp(-s / r_s))


def flat_newtonian_model(s):
    """Flat Newtonian model: v_tilde = 1 everywhere."""
    return np.ones_like(np.asarray(s), dtype=float)


def constant_boost_model(s, alpha):
    """Constant boost model: v_tilde = 1 + alpha everywhere."""
    return np.ones_like(np.asarray(s), dtype=float) * (1.0 + alpha)


def chi2_statistic(observed, expected, errors):
    """Diagonal chi-squared statistic with per-bin Gaussian errors."""
    return float(np.sum(((observed - expected) / errors) ** 2))

def temporal_topology_suppression(rho, rho_c=RHO_CRIT_G_CM3, kappa_bare=KAPPA_GAL):
    """Continuous Temporal Topology suppression factor."""
    x = np.log10(rho / rho_c) / 0.5
    suppression = 1.0 / (1.0 + np.exp(x))
    return kappa_bare * suppression

# =============================================================================
# 3b. PROFILE BUILDING UTILITIES (Wide Binary Sector)
# =============================================================================

def build_profile(frame, value_col="v_tilde", bins=GLOBAL_BINS, n_baseline_bins=5):
    """
    Build a binned, baseline-normalized velocity profile from a dataframe.

    Parameters
    ----------
    frame : pd.DataFrame
        Dataframe containing sep_AU and velocity column
    value_col : str
        Column name for velocity values (default: "v_tilde")
    bins : array
        Bin edges for separation (default: GLOBAL_BINS)
    n_baseline_bins : int
        Number of inner bins to use for baseline normalization (default: 5)

    Returns
    -------
    profile : pd.DataFrame or None
        Binned profile with normalized velocities, or None if insufficient data
    """
    work = frame[["sep_AU", value_col]].dropna().copy()
    if len(work) == 0:
        return None

    work["bin"] = pd.cut(work["sep_AU"], bins=bins)
    grouped = work.groupby("bin", observed=True)[value_col]
    bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
    bin_index = work["bin"].cat.categories

    profile = pd.DataFrame({
        "sep_AU": bin_centers,
        "v_tilde_median": grouped.median().reindex(bin_index).to_numpy(),
        "v_tilde_std": grouped.std().reindex(bin_index).to_numpy(),
        "count": grouped.count().reindex(bin_index).to_numpy(),
    }).dropna()

    if len(profile) < 6:
        return None

    # Standard Error of the Median (1.253 = sqrt(pi/2) for Gaussian)
    profile["sem"] = 1.253 * profile["v_tilde_std"] / np.sqrt(profile["count"])

    # Baseline normalization using inner bins
    end_idx = min(n_baseline_bins, len(profile))
    baseline = profile["v_tilde_median"].iloc[:end_idx].mean()
    if not np.isfinite(baseline) or baseline <= 0:
        return None

    profile["v_tilde_norm"] = profile["v_tilde_median"] / baseline
    profile["sem_norm"] = profile["sem"] / baseline
    profile = profile.replace([np.inf, -np.inf], np.nan).dropna()

    return profile


# =============================================================================
# 4. MASS CORRECTIONS & BIAS
# =============================================================================

def isochrony_mass_bias(gamma_t, n=ALPHA_NUCLEAR):
    """M/L ratio bias: M_obs / M_true = Gamma_t^n."""
    return np.power(np.maximum(gamma_t, 0.01), n)

def correct_stellar_mass(log_Mstar, gamma_t, n=ALPHA_NUCLEAR):
    """Apply TEP correction to observed stellar mass."""
    return log_Mstar - np.log10(isochrony_mass_bias(gamma_t, n))

def stellar_to_halo_mass_behroozi_like(log_Mstar, z):
    """Empirical SMHM relation proxy for high-z."""
    log_ratio = -1.8 - 0.1 * (log_Mstar - 10) + 0.05 * (z - 5)
    return log_Mstar - log_ratio

# =============================================================================
# 5. COSMOLOGY UTILS
# =============================================================================

def cosmic_time_gyr(z, H0=67.4, Om=0.315):
    """Cosmic time at redshift z in Gyr (flat LCDM)."""
    z = np.atleast_1d(z)
    H0_s = H0 * 1e3 / 3.0857e22
    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(Om * (1 + zp)**3 + (1-Om)))
    results = [integrate.quad(integrand, zi, np.inf)[0] / H0_s / 3.156e16 for zi in z]
    return np.array(results) if len(results) > 1 else results[0]
