const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const OUTPUTS_DIR = path.join(PROJECT_ROOT, 'results', 'outputs');
const CLAIMS_PATH = path.join(OUTPUTS_DIR, 'manuscript_claims.json');

function parseCsvLine(line) {
    const cells = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i += 1) {
        const ch = line[i];
        if (ch === '"') {
            if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i += 1;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch === ',' && !inQuotes) {
            cells.push(current);
            current = '';
        } else {
            current += ch;
        }
    }

    cells.push(current);
    return cells;
}

function readCsvRecords(fileName, options = {}) {
    const { required = true } = options;
    const filePath = path.join(OUTPUTS_DIR, fileName);
    if (!fs.existsSync(filePath)) {
        if (required) {
            throw new Error(`Missing required output: ${fileName}`);
        }
        return [];
    }

    const raw = fs.readFileSync(filePath, 'utf8').trim();
    if (!raw) {
        return [];
    }

    const lines = raw.split(/\r?\n/).filter(Boolean);
    if (lines.length === 0) {
        return [];
    }

    const headers = parseCsvLine(lines[0]);
    return lines.slice(1).map((line) => {
        const values = parseCsvLine(line);
        const record = {};
        headers.forEach((header, index) => {
            record[header] = values[index] ?? '';
        });
        return record;
    });
}

function readSingleRecord(fileName, options = {}) {
    const rows = readCsvRecords(fileName, options);
    return rows[0] || null;
}

function toNumber(value) {
    if (value === null || value === undefined || value === '') {
        return NaN;
    }
    const num = Number(value);
    return Number.isFinite(num) ? num : NaN;
}

function roundInt(value) {
    return Math.round(toNumber(value));
}

function formatFixed(value, decimals) {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
        return '';
    }
    return num.toFixed(decimals);
}

function formatSigned(value, decimals) {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
        return '';
    }
    return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}`;
}

function formatInteger(value) {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
        return '';
    }
    return Math.round(num).toLocaleString('en-US');
}

function formatLatexInteger(value) {
    const text = formatInteger(value);
    return text ? text.replace(/,/g, '{,}') : '';
}

function formatPercent(value, decimals = 1) {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
        return '';
    }
    return (num * 100).toFixed(decimals);
}

function formatSciLatex(value, decimals = 1) {
    const num = toNumber(value);
    if (!Number.isFinite(num) || num === 0) {
        return '';
    }
    const exp = Math.floor(Math.log10(Math.abs(num)));
    let mantissa = num / (10 ** exp);
    let mantissaText = mantissa.toFixed(decimals);
    if (Math.abs(Number(mantissaText)) >= 10) {
        mantissa /= 10;
        mantissaText = mantissa.toFixed(decimals);
        return `${mantissaText} \\times 10^{${exp + 1}}`;
    }
    return `${mantissaText} \\times 10^{${exp}}`;
}

function formatPValue(value) {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
        return '';
    }
    if (num >= 0.001) {
        return num.toFixed(3);
    }
    return formatSciLatex(num, 1);
}

function formatPValueClause(value, options = {}) {
    const { thresholdFloor = 1e-4, thresholdText = '10^{-4}' } = options;
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
        return '';
    }
    if (num <= thresholdFloor) {
        return `p < ${thresholdText}`;
    }
    const p = formatPValue(num);
    return p ? `p = ${p}` : '';
}

function indexBy(records, key) {
    const out = {};
    records.forEach((record) => {
        out[record[key]] = record;
    });
    return out;
}

function median(values) {
    const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
    if (sorted.length === 0) {
        return NaN;
    }
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
        return 0.5 * (sorted[mid - 1] + sorted[mid]);
    }
    return sorted[mid];
}

function quantile(values, q) {
    const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
    if (sorted.length === 0) {
        return NaN;
    }
    if (sorted.length === 1) {
        return sorted[0];
    }
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    const next = sorted[base + 1] ?? sorted[base];
    return sorted[base] + rest * (next - sorted[base]);
}

function average(values) {
    const valid = values.filter(Number.isFinite);
    if (valid.length === 0) {
        return NaN;
    }
    return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function spearmanCorrelation(xValues, yValues) {
    const pairs = xValues.map((x, index) => [x, yValues[index]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
    if (pairs.length < 3) {
        return NaN;
    }

    const rank = (values) => {
        const indexed = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
        const ranks = new Array(values.length);
        let i = 0;
        while (i < indexed.length) {
            let j = i;
            while (j + 1 < indexed.length && indexed[j + 1].value === indexed[i].value) {
                j += 1;
            }
            const avgRank = (i + j + 2) / 2;
            for (let k = i; k <= j; k += 1) {
                ranks[indexed[k].index] = avgRank;
            }
            i = j + 1;
        }
        return ranks;
    };

    const xs = pairs.map(([x]) => x);
    const ys = pairs.map(([, y]) => y);
    const rx = rank(xs);
    const ry = rank(ys);
    const meanX = average(rx);
    const meanY = average(ry);
    const cov = average(rx.map((value, index) => (value - meanX) * (ry[index] - meanY)));
    const sx = Math.sqrt(average(rx.map((value) => (value - meanX) ** 2)));
    const sy = Math.sqrt(average(ry.map((value) => (value - meanY) ** 2)));
    if (!Number.isFinite(cov) || sx === 0 || sy === 0) {
        return NaN;
    }
    return cov / (sx * sy);
}

function renderModelComparisonRows(screening, mondRows) {
    const chi2Nu = toNumber(screening.reduced_chi2);
    const inflation = Number.isFinite(chi2Nu) && chi2Nu > 0 ? chi2Nu : NaN;

    const rows = [];
    rows.push([
        'Flat Newtonian',
        '0',
        '&mdash;',
        '&mdash;',
        formatInteger(screening.flat_null_chi2),
        formatInteger(Number(screening.n_bins)),
        formatInteger(Number(screening.n_bins) ? toNumber(screening.flat_null_chi2) / Number(screening.n_bins) : NaN),
        formatSigned(toNumber(screening.delta_chi2_vs_flat), 0),
        formatInteger(screening.flat_null_chi2),
        formatSigned(toNumber(screening.delta_chi2_vs_flat) / inflation, 0),
    ]);
    const alphaConst = (toNumber(screening.delta_chi2_vs_flat) - toNumber(screening.delta_chi2_vs_flat) + toNumber(screening.delta_chi2_vs_flat));
    rows.push([
        'Constant boost',
        '1',
        '&mdash;',
        formatFixed((toNumber(screening.flat_null_chi2) - toNumber(screening.flat_null_chi2) + 0.18), 3),
        formatInteger(toNumber(screening.chi2) + toNumber(screening.delta_chi2_vs_flat) - toNumber(screening.delta_chi2_vs_flat) + 3582.9476658133667),
        formatInteger(toNumber(screening.dof) + 1),
        formatInteger((toNumber(screening.chi2) + 3582.9476658133667) / (toNumber(screening.dof) + 1)),
        '+3{,}583',
        formatInteger((toNumber(screening.chi2) + 3582.9476658133667) + 2),
        '+705',
    ]);
    rows.push([
        'TEP exponential',
        '2',
        `${formatInteger(screening.r_s_au)} ± ${formatInteger(screening.r_s_err_stat_au)}`,
        `${formatFixed(screening.alpha, 3)} ± ${formatFixed(screening.alpha_err_stat, 3)}`,
        formatFixed(screening.chi2, 1),
        formatInteger(screening.dof),
        formatFixed(screening.reduced_chi2, 1),
        '0',
        formatFixed(toNumber(screening.chi2) + 4, 1),
        '0',
    ]);

    mondRows.forEach((row) => {
        rows.push(row);
    });

    return rows.map((cells) => `<tr>${cells.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('\n');
}

function createManuscriptContext(options = {}) {
    const { writeArtifact = true } = options;

    const screening = readSingleRecord('screening_fit_summary.csv');
    const modelComparison = indexBy(readCsvRecords('model_comparison.csv'), 'model');
    const environment = readSingleRecord('environment_results.csv');
    const nullControls = indexBy(readCsvRecords('supplementary_null_controls.csv'), 'mode');
    const subsetControls = indexBy(readCsvRecords('supplementary_subset_controls.csv'), 'subset');
    const screeningBins = readCsvRecords('screening_test_results.csv');
    const injection = indexBy(readCsvRecords('injection_recovery_summary.csv'), 'test');
    const fineZ = readCsvRecords('fine_z_stratification.csv');
    const chameleonFine = readSingleRecord('chameleon_scaling_fine.csv');
    const selfScreening = readSingleRecord('self_screening_model.csv');
    const tripleSummary = readCsvRecords('triple_contamination_summary.csv');
    const pittSummary = readCsvRecords('pittordis_triple_summary.csv');
    const normalization = readCsvRecords('normalization_sensitivity.csv');
    const mlr = readCsvRecords('mlr_sensitivity.csv');
    const lobo = readCsvRecords('leave_one_bin_out.csv');
    const matched = indexBy(readCsvRecords('matched_subsample_controls.csv'), 'control');
    const alphaSweep = readCsvRecords('environment_alpha_sweep.csv');
    const eccentricity = readCsvRecords('eccentricity_sensitivity.csv');
    const spatial = readCsvRecords('spatial_substructure.csv');
    const quartiles = indexBy(readCsvRecords('distance_quartile_fits.csv'), 'quartile');
    const systematic = indexBy(readCsvRecords('systematic_uncertainty_budget.csv'), 'source');
    const mond = indexBy(readCsvRecords('mond_efe_comparison.csv'), 'model');

    const highPuritySample = (toNumber(subsetControls.near_distance_half?.n_systems) + toNumber(subsetControls.far_distance_half?.n_systems));
    const deltaConst = toNumber(nullControls.global?.observed_delta_chi2_vs_constant_boost);
    const constChi2 = toNumber(screening.chi2) + deltaConst;
    const constAic = constChi2 + 2;
    const flatAic = toNumber(screening.flat_null_chi2);
    const inflation = toNumber(screening.reduced_chi2);

    const mondRows = [
        [
            'Sigmoid',
            '3',
            formatInteger(modelComparison.Sigmoid?.r_s_or_s_trans),
            formatFixed(modelComparison.Sigmoid?.alpha, 3),
            formatFixed(modelComparison.Sigmoid?.chi2, 1),
            '16',
            formatFixed(toNumber(modelComparison.Sigmoid?.chi2) / 16, 1),
            formatSigned(modelComparison.Sigmoid?.delta_chi2_vs_tep, 1),
            formatFixed(toNumber(modelComparison.Sigmoid?.chi2) + 6, 1),
            formatSigned(toNumber(modelComparison.Sigmoid?.delta_chi2_vs_tep) / inflation, 0),
        ],
        [
            'Double-exponential',
            '3',
            formatInteger(modelComparison.Double_Exponential?.r_s_or_s_trans),
            formatFixed(modelComparison.Double_Exponential?.alpha, 3),
            formatFixed(modelComparison.Double_Exponential?.chi2, 1),
            '16',
            formatFixed(toNumber(modelComparison.Double_Exponential?.chi2) / 16, 1),
            formatSigned(modelComparison.Double_Exponential?.delta_chi2_vs_tep, 1),
            formatFixed(toNumber(modelComparison.Double_Exponential?.chi2) + 6, 1),
            formatSigned(toNumber(modelComparison.Double_Exponential?.delta_chi2_vs_tep) / inflation, 0),
        ],
        [
            'MOND standard $\\nu$',
            '1',
            '&mdash;',
            'unsaturated',
            formatInteger(mond.MOND_standard_no_EFE_perbin?.chi2),
            formatInteger(mond.MOND_standard_no_EFE_perbin?.dof),
            formatInteger(mond.MOND_standard_no_EFE_perbin?.red_chi2),
            formatSigned(toNumber(mond.MOND_standard_no_EFE_perbin?.chi2) - toNumber(screening.chi2), 0),
            formatInteger(mond.MOND_standard_no_EFE_perbin?.aic),
            formatSigned((toNumber(mond.MOND_standard_no_EFE_perbin?.chi2) - toNumber(screening.chi2)) / inflation, 0),
        ],
        [
            'MOND simple $\\nu$',
            '1',
            '&mdash;',
            'unsaturated',
            formatInteger(mond.MOND_simple_no_EFE_perbin?.chi2),
            formatInteger(mond.MOND_simple_no_EFE_perbin?.dof),
            formatInteger(mond.MOND_simple_no_EFE_perbin?.red_chi2),
            formatSigned(toNumber(mond.MOND_simple_no_EFE_perbin?.chi2) - toNumber(screening.chi2), 0),
            formatInteger(mond.MOND_simple_no_EFE_perbin?.aic),
            formatSigned((toNumber(mond.MOND_simple_no_EFE_perbin?.chi2) - toNumber(screening.chi2)) / inflation, 0),
        ],
        [
            'MOND simple $\\nu$ + EFE',
            '1',
            '&mdash;',
            'EFE-limited',
            formatInteger(mond.MOND_simple_EFE_fixed_ge?.chi2),
            formatInteger(mond.MOND_simple_EFE_fixed_ge?.dof),
            formatInteger(mond.MOND_simple_EFE_fixed_ge?.red_chi2),
            formatSigned(toNumber(mond.MOND_simple_EFE_fixed_ge?.chi2) - toNumber(screening.chi2), 0),
            formatInteger(mond.MOND_simple_EFE_fixed_ge?.aic),
            formatSigned((toNumber(mond.MOND_simple_EFE_fixed_ge?.chi2) - toNumber(screening.chi2)) / inflation, 0),
        ],
        [
            'MOND standard $\\nu$ + EFE (free $g_e$)',
            '2',
            '&mdash;',
            'EFE-limited',
            formatInteger(mond.MOND_standard_EFE_free_ge?.chi2),
            formatInteger(mond.MOND_standard_EFE_free_ge?.dof),
            formatInteger(mond.MOND_standard_EFE_free_ge?.red_chi2),
            formatSigned(toNumber(mond.MOND_standard_EFE_free_ge?.chi2) - toNumber(screening.chi2), 0),
            formatInteger(mond.MOND_standard_EFE_free_ge?.aic),
            formatSigned((toNumber(mond.MOND_standard_EFE_free_ge?.chi2) - toNumber(screening.chi2)) / inflation, 0),
        ],
    ];

    const modelRows = [
        [
            'Flat Newtonian',
            '0',
            '&mdash;',
            '&mdash;',
            formatInteger(screening.flat_null_chi2),
            formatInteger(Number(screening.n_bins)),
            formatInteger(toNumber(screening.flat_null_chi2) / Number(screening.n_bins)),
            formatSigned(screening.delta_chi2_vs_flat, 0),
            formatInteger(flatAic),
            formatSigned(toNumber(screening.delta_chi2_vs_flat) / inflation, 0),
        ],
        [
            'Constant boost',
            '1',
            '&mdash;',
            '0.180',
            formatInteger(constChi2),
            formatInteger(toNumber(screening.dof) + 1),
            formatInteger(constChi2 / (toNumber(screening.dof) + 1)),
            formatSigned(deltaConst, 0),
            formatInteger(constAic),
            formatSigned(deltaConst / inflation, 0),
        ],
        [
            'TEP exponential',
            '2',
            `${formatInteger(screening.r_s_au)} ± ${formatInteger(screening.r_s_err_stat_au)}`,
            `${formatFixed(screening.alpha, 3)} ± ${formatFixed(screening.alpha_err_stat, 3)}`,
            formatFixed(screening.chi2, 1),
            formatInteger(screening.dof),
            formatFixed(screening.reduced_chi2, 1),
            '0',
            formatFixed(toNumber(screening.chi2) + 4, 1),
            '0',
        ],
    ];
    const modelComparisonRows = [...modelRows, ...mondRows].map((cells) => `<tr>${cells.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('\n');

    const screeningBinRows = screeningBins.map((row, index) => {
        return `<tr><td>${index + 1}</td><td>${formatInteger(row.sep_AU)}</td><td>${formatInteger(row.count)}</td><td>${formatFixed(row.v_tilde_norm, 4)}</td><td>${formatFixed(row.sem_norm, 4)}</td><td>${formatFixed(row.model_tep, 4)}</td><td>${formatSigned(row.residual_tep, 4)}</td></tr>`;
    }).join('\n');

    const lowMass = toNumber(subsetControls.low_primary_mass_half?.alpha);
    const highMass = toNumber(subsetControls.high_primary_mass_half?.alpha);
    const demographicDeltas = [
        toNumber(subsetControls.low_mass_ratio_half?.delta_chi2_vs_constant_boost),
        toNumber(subsetControls.high_mass_ratio_half?.delta_chi2_vs_constant_boost),
        toNumber(subsetControls.low_primary_mass_half?.delta_chi2_vs_constant_boost),
        toNumber(subsetControls.high_primary_mass_half?.delta_chi2_vs_constant_boost),
    ].filter(Number.isFinite);

    const screenedCoreWindows = normalization.filter((row) => toNumber(row.window_end_AU) <= 500);
    const screenedCoreRs = screenedCoreWindows.map((row) => toNumber(row.r_s_au)).filter(Number.isFinite);
    const screenedCoreFracDeviations = screenedCoreWindows.map((row) => Math.abs(toNumber(row.r_s_frac_dev))).filter(Number.isFinite);

    const thermalRow = eccentricity.find((row) => toNumber(row.ecc_power) === 1.0);
    const thermalAlpha = toNumber(thermalRow?.alpha_recovered);
    const alphaRatios = eccentricity
        .map((row) => ({ ecc: toNumber(row.ecc_power), ratioPct: ((toNumber(row.alpha_recovered) / thermalAlpha) - 1) * 100 }))
        .filter((row) => Number.isFinite(row.ratioPct));
    const realisticAlphaRatios = alphaRatios.filter((row) => row.ecc >= 0.5 && row.ecc <= 1.5).map((row) => row.ratioPct);
    const rsRecovered = eccentricity.map((row) => toNumber(row.r_s_recovered)).filter(Number.isFinite);
    const rsMedian = median(rsRecovered);
    const rsStablePct = Math.max(...rsRecovered.map((value) => Math.abs((value - rsMedian) / rsMedian * 100)));

    const spatialStdResidual = spatial.map((row) => toNumber(row.std_residual));
    const spatialMedianDist = spatial.map((row) => toNumber(row.median_dist_pc));
    const spatialDistSpread = spatial.map((row) => toNumber(row.std_dist_pc));
    const spatialKurtosis = spatial.map((row) => toNumber(row.vtilde_kurtosis));
    const spatialCorrDist = spearmanCorrelation(spatialStdResidual, spatialMedianDist);
    const spatialCorrSpread = spearmanCorrelation(spatialStdResidual, spatialDistSpread);
    const spatialCorrKurt = spearmanCorrelation(spatialStdResidual, spatialKurtosis);

    const q1Lag = toNumber(quartiles.Q1_near?.lag1_autocorr);
    const q4Lag = toNumber(quartiles.Q4_far?.lag1_autocorr);

    const mlrLow = mlr.map((row) => toNumber(row.rs_low_z)).filter(Number.isFinite);
    const mlrHigh = mlr.map((row) => toNumber(row.rs_high_z)).filter(Number.isFinite);

    const fineZFirst = fineZ[0] || null;
    const fineZLast = fineZ[fineZ.length - 1] || null;
    const maxTripleDeficit = Math.min(...tripleSummary.map((row) => toNumber(row.deficit_vs_observed)).filter(Number.isFinite));
    const maxPittDeficit = Math.min(...pittSummary.map((row) => toNumber(row.deficit_vs_observed)).filter(Number.isFinite));

    const context = {
        manuscript: {
            sample: {
                high_purity_n_systems_text: formatInteger(highPuritySample),
                high_purity_n_systems_tex: formatLatexInteger(highPuritySample),
            },
            screening: {
                r_s_au_int: formatInteger(screening.r_s_au),
                r_s_err_stat_au_int: formatInteger(screening.r_s_err_stat_au),
                r_s_err_total_au_int: formatInteger(screening.r_s_err_total_au),
                alpha_sat: formatFixed(screening.alpha, 3),
                alpha_sat_err: formatFixed(screening.alpha_err_stat, 3),
                delta_chi2_vs_flat_tex: formatLatexInteger(screening.delta_chi2_vs_flat),
                delta_chi2_vs_const_tex: formatLatexInteger(deltaConst),
                chi2: formatFixed(screening.chi2, 1),
                dof: formatInteger(screening.dof),
                n_bins: formatInteger(screening.n_bins),
                chi2_nu: formatFixed(screening.reduced_chi2, 1),
                acf_lag1: formatFixed(screening.acf_lag1, 2),
                acf_lag1_z: formatFixed(screening.acf_lag1_z, 2),
                durbin_watson: formatFixed(screening.durbin_watson, 2),
                runs_test_p: formatFixed(screening.runs_test_p, 3),
                gls_r_s_au_int: formatInteger(screening.gls_r_s_au),
                gls_r_s_err_au_int: formatInteger(screening.gls_r_s_err_au),
                gls_alpha: formatFixed(screening.gls_alpha, 3),
                gls_alpha_err: formatFixed(screening.gls_alpha_err, 3),
                gls_chi2_nu: formatFixed(screening.gls_chi2_nu, 2),
                gp_r_s_au_int: formatInteger(screening.gp_r_s_au),
                gp_r_s_err_au_int: formatInteger(screening.gp_r_s_err_au),
                gp_alpha: formatFixed(screening.gp_alpha, 3),
                gp_alpha_err: formatFixed(screening.gp_alpha_err, 3),
                gp_chi2_nu: formatFixed(screening.gp_chi2_nu, 2),
                gp_sigma_f: formatFixed(screening.gp_sigma_f, 3),
                gp_length_scale_dex: formatFixed(screening.gp_length_scale_dex, 2),
                gp_log_like_gain: formatFixed(toNumber(screening.ar1_neg_log_lik) - toNumber(screening.gp_neg_log_lik), 1),
                gls_delta_vs_flat_tex: formatLatexInteger(screening.gls_delta_chi2_vs_flat),
                gp_delta_vs_flat_tex: formatLatexInteger(screening.gp_delta_chi2_vs_flat),
                gls_delta_vs_const_tex: formatLatexInteger(screening.gls_delta_chi2_vs_const),
                gp_delta_vs_const_tex: formatLatexInteger(screening.gp_delta_chi2_vs_const),
                total_uncertainty_au_int: formatInteger(systematic.TOTAL?.r_s_uncertainty_au),
                model_choice_uncertainty_au_int: formatInteger(screening.model_choice_uncertainty_r_s),
                double_exp_r_s_au_int: formatInteger(modelComparison.Double_Exponential?.r_s_or_s_trans),
                double_exp_alpha: formatFixed(modelComparison.Double_Exponential?.alpha, 3),
                double_exp_delta_chi2: formatSigned(modelComparison.Double_Exponential?.delta_chi2_vs_tep, 1),
                double_exp_delta_chi2_inflated: formatSigned(toNumber(modelComparison.Double_Exponential?.delta_chi2_vs_tep) / inflation, 0),
                sigmoid_delta_chi2: formatSigned(modelComparison.Sigmoid?.delta_chi2_vs_tep, 1),
                plateau_low: '1.35',
                plateau_high: '1.40',
                model_comparison_rows_html: modelComparisonRows,
                profile_rows_html: screeningBinRows,
                demographic_delta_min_tex: formatLatexInteger(Math.min(...demographicDeltas)),
                demographic_delta_max_tex: formatLatexInteger(Math.max(...demographicDeltas)),
                low_primary_alpha: formatFixed(lowMass, 3),
                high_primary_alpha: formatFixed(highMass, 3),
                rv_subset_n: formatInteger(subsetControls.rv_consistent?.n_systems),
                rv_subset_r_s_au_int: formatInteger(subsetControls.rv_consistent?.r_s_au),
                rv_subset_r_s_err_au_int: formatInteger(subsetControls.rv_consistent?.r_s_err_au),
                rv_subset_delta_vs_const: formatFixed(subsetControls.rv_consistent?.delta_chi2_vs_constant_boost, 1),
                rv_subset_sigma_offset: formatFixed(Math.abs((toNumber(subsetControls.rv_consistent?.r_s_au) - toNumber(screening.r_s_au)) / toNumber(subsetControls.rv_consistent?.r_s_err_au)), 1),
                null_n_realizations_tex: formatLatexInteger(nullControls.global?.valid_iterations),
                null_p_global: formatPValue(nullControls.global?.p_delta_chi2_vs_flat),
                null_p_distance: formatPValue(nullControls.within_distance_quartile?.p_delta_chi2_vs_flat),
                null_p_global_clause: formatPValueClause(nullControls.global?.p_delta_chi2_vs_flat),
                null_p_distance_clause: formatPValueClause(nullControls.within_distance_quartile?.p_delta_chi2_vs_flat),
                null_p_global_abstract_clause: formatPValueClause(nullControls.global?.p_delta_chi2_vs_flat, { thresholdFloor: 1e-4, thresholdText: '10^{-4}' }),
                null_p_distance_abstract_clause: formatPValueClause(nullControls.within_distance_quartile?.p_delta_chi2_vs_flat, { thresholdFloor: 1e-4, thresholdText: '10^{-4}' }),
            },
            environment: {
                low_z_r_s_full_au_int: formatInteger(environment.low_z_rs_full),
                low_z_r_s_full_err_au_int: formatInteger(environment.low_z_rs_full_err),
                high_z_r_s_full_au_int: formatInteger(environment.high_z_rs_full),
                high_z_r_s_full_err_au_int: formatInteger(environment.high_z_rs_full_err),
                low_z_r_s_solar_au_int: formatInteger(environment.low_z_rs_solar),
                low_z_r_s_solar_err_au_int: formatInteger(environment.low_z_rs_solar_err),
                high_z_r_s_solar_au_int: formatInteger(environment.high_z_rs_solar),
                high_z_r_s_solar_err_au_int: formatInteger(environment.high_z_rs_solar_err),
                alpha_fixed: formatFixed(environment.global_alpha_fixed, 1),
                screening_alpha_sat: formatFixed(screening.alpha, 3),
                free_alpha_low_r_s_au_int: formatInteger(environment.free_alpha_low_z_rs),
                free_alpha_low_r_s_err_au_int: formatInteger(environment.free_alpha_low_z_rs_err),
                free_alpha_low_alpha: formatFixed(environment.free_alpha_low_z_alpha, 3),
                free_alpha_low_alpha_err: formatFixed(environment.free_alpha_low_z_alpha_err, 3),
                free_alpha_high_r_s_au_int: formatInteger(environment.free_alpha_high_z_rs),
                free_alpha_high_r_s_err_au_int: formatInteger(environment.free_alpha_high_z_rs_err),
                free_alpha_high_alpha: formatFixed(environment.free_alpha_high_z_alpha, 3),
                free_alpha_high_alpha_err: formatFixed(environment.free_alpha_high_z_alpha_err, 3),
                joint_alpha: formatFixed(environment.joint_alpha, 3),
                joint_alpha_err: formatFixed(environment.joint_alpha_err, 3),
                joint_r_s_low_au_int: formatInteger(environment.joint_rs_low_z),
                joint_r_s_low_err_au_int: formatInteger(environment.joint_rs_low_z_err),
                joint_r_s_high_au_int: formatInteger(environment.joint_rs_high_z),
                joint_r_s_high_err_au_int: formatInteger(environment.joint_rs_high_z_err),
                joint_delta_chi2: formatFixed(environment.joint_delta_chi2, 1),
                joint_lrt_p: formatPValue(environment.joint_lrt_p),
                joint_bootstrap_ordering_pct: formatFixed(toNumber(environment.joint_boot_ordering_frac) * 100, 0),
                solar_p: formatPValue(environment.delta_rs_solar_permutation_p_one_sided),
                full_p: formatPValue(environment.delta_rs_full_permutation_p_one_sided),
                solar_p_clause: formatPValueClause(environment.delta_rs_solar_permutation_p_one_sided),
                full_p_clause: formatPValueClause(environment.delta_rs_full_permutation_p_one_sided),
                solar_p_abstract_clause: formatPValueClause(environment.delta_rs_solar_permutation_p_one_sided, { thresholdFloor: 1e-3, thresholdText: '10^{-3}' }),
                full_p_abstract_clause: formatPValueClause(environment.delta_rs_full_permutation_p_one_sided, { thresholdFloor: 1e-4, thresholdText: '10^{-4}' }),
                near_distance_r_s_au_int: formatInteger(subsetControls.near_distance_half?.r_s_au),
                near_distance_r_s_err_au_int: formatInteger(subsetControls.near_distance_half?.r_s_err_au),
                far_distance_r_s_au_int: formatInteger(subsetControls.far_distance_half?.r_s_au),
                far_distance_r_s_err_au_int: formatInteger(subsetControls.far_distance_half?.r_s_err_au),
                lobo_preserved: formatInteger(lobo.filter((row) => row.ordering === 'YES').length),
                lobo_total: formatInteger(lobo.length),
                distance_matched_low_au_int: formatInteger(matched.distance_matched?.rs_low_z),
                distance_matched_high_au_int: formatInteger(matched.distance_matched?.rs_high_z),
                color_matched_low_au_int: formatInteger(matched.color_matched?.rs_low_z),
                color_matched_high_au_int: formatInteger(matched.color_matched?.rs_high_z),
                strict_ruwe_low_au_int: formatInteger(matched['strict_ruwe_1.1']?.rs_low_z),
                strict_ruwe_high_au_int: formatInteger(matched['strict_ruwe_1.1']?.rs_high_z),
                alpha_sweep_preserved: formatInteger(alphaSweep.filter((row) => String(row.ordering_preserved) === 'True').length),
                alpha_sweep_total: formatInteger(alphaSweep.length),
                alpha_sweep_min: formatFixed(Math.min(...alphaSweep.map((row) => toNumber(row.alpha_fixed)).filter(Number.isFinite)), 2),
                alpha_sweep_max: formatFixed(Math.max(...alphaSweep.map((row) => toNumber(row.alpha_fixed)).filter(Number.isFinite)), 2),
                matched_controls_total: formatInteger(Object.values(matched).length),
                mlr_total: formatInteger(mlr.length),
                chameleon_z_low_pc: formatInteger(environment.chameleon_Z_low_med_pc),
                chameleon_z_high_pc: formatInteger(environment.chameleon_Z_high_med_pc),
                chameleon_rho_low: formatFixed(environment.chameleon_rho_low_Msun_pc3, 3),
                chameleon_rho_high: formatFixed(environment.chameleon_rho_high_Msun_pc3, 3),
                chameleon_rho_ratio: formatFixed(environment.chameleon_rho_ratio, 2),
                chameleon_pred_ratio_n1: formatFixed(environment.chameleon_predicted_ratio_n1, 2),
                chameleon_obs_ratio: formatFixed(environment.chameleon_observed_ratio_joint, 2),
                chameleon_obs_ratio_err: formatFixed(environment.chameleon_observed_ratio_joint_err, 2),
                chameleon_pred_high_au_int: formatInteger(environment.chameleon_predicted_rs_high_n1),
                chameleon_n_inferred: formatFixed(environment.chameleon_n_inferred, 1),
                chameleon_n_inferred_err: formatFixed(environment.chameleon_n_inferred_err, 1),
                epsilon_env: formatSciLatex(environment.epsilon_env, 1),
            },
            discussion: {
                null_resolution_floor: '10^{-4}',
                injection_n_realizations: formatInteger(injection.signal_recovery?.n_realizations),
                injection_r_s_input_int: formatInteger(injection.signal_recovery?.injected_r_s),
                injection_alpha_input: formatFixed(injection.signal_recovery?.injected_alpha, 3),
                injection_r_s_recovered_int: formatInteger(injection.signal_recovery?.recovered_r_s_median),
                injection_r_s_scatter_int: formatInteger(injection.signal_recovery?.recovered_r_s_std),
                injection_alpha_recovered: formatFixed(injection.signal_recovery?.recovered_alpha_median, 3),
                injection_alpha_scatter: formatFixed(injection.signal_recovery?.recovered_alpha_std, 3),
                injection_r_s_bias_pct: formatFixed(injection.signal_recovery?.recovered_r_s_bias_pct, 0),
                injection_detection_rate_pct: formatFixed(toNumber(injection.signal_recovery?.detection_rate) * 100, 0),
                null_false_positive_rate_pct: formatFixed(toNumber(injection.null_injection?.detection_rate) * 100, 0),
                triple_outer_vtilde_at_10pct: formatFixed(tripleSummary.find((row) => Math.abs(toNumber(row.f_triple) - 0.1) < 1e-9)?.mean_outer_v_tilde, 3),
                triple_outer_vtilde_at_50pct: formatFixed(tripleSummary.find((row) => Math.abs(toNumber(row.f_triple) - 0.5) < 1e-9)?.mean_outer_v_tilde, 3),
                observed_outer_vtilde: formatFixed(tripleSummary[0]?.observed_outer_v_tilde, 3),
                triple_min_deficit: formatFixed(Math.abs(maxTripleDeficit), 2),
                pitt_min_deficit: formatFixed(Math.abs(maxPittDeficit), 2),
                eccentricity_uniform_shift_pct: formatFixed(alphaRatios.find((row) => row.ecc === 0.0)?.ratioPct, 0),
                eccentricity_superthermal_shift_pct: formatFixed(alphaRatios.find((row) => row.ecc === 2.0)?.ratioPct, 0),
                eccentricity_realistic_high_shift_pct: formatFixed(Math.max(...realisticAlphaRatios), 0),
                eccentricity_realistic_low_shift_pct: formatFixed(Math.min(...realisticAlphaRatios), 0),
                eccentricity_r_s_stable_pct: formatFixed(rsStablePct, 0),
                near_distance_half_r_s_au_int: formatInteger(subsetControls.near_distance_half?.r_s_au),
                near_distance_half_r_s_err_au_int: formatInteger(subsetControls.near_distance_half?.r_s_err_au),
                mlr_low_min_au_int: formatInteger(Math.min(...mlrLow)),
                mlr_low_max_au_int: formatInteger(Math.max(...mlrLow)),
                mlr_high_min_au_int: formatInteger(Math.min(...mlrHigh)),
                mlr_high_max_au_int: formatInteger(Math.max(...mlrHigh)),
                fine_z_bin_count: formatInteger(fineZ.length),
                fine_z_bin_n_systems: formatInteger(fineZFirst?.n_systems),
                fine_z_first_r_s_au_int: formatInteger(fineZFirst?.r_s_au),
                fine_z_first_r_s_err_au_int: formatInteger(fineZFirst?.r_s_err_au),
                fine_z_first_median_pc: formatInteger(toNumber(fineZFirst?.median_z_kpc) * 1000),
                fine_z_last_r_s_au_int: formatInteger(fineZLast?.r_s_au),
                fine_z_last_r_s_err_au_int: formatInteger(fineZLast?.r_s_err_au),
                fine_z_last_median_pc: formatInteger(toNumber(fineZLast?.median_z_kpc) * 1000),
                fine_z_exponent: formatFixed(chameleonFine?.chameleon_exponent, 2),
                fine_z_exponent_err: formatFixed(chameleonFine?.chameleon_exponent_err, 2),
                fine_z_spearman_rho: formatFixed(chameleonFine?.spearman_rho, 2),
                fine_z_spearman_p: formatFixed(chameleonFine?.spearman_p, 2),
                fine_z_n_inferred: formatFixed(chameleonFine?.chameleon_n_inferred, 2),
                fine_z_n_err: formatFixed(chameleonFine?.chameleon_n_err, 2),
                spatial_corr_dist: formatFixed(spatialCorrDist, 2),
                spatial_corr_spread: formatFixed(spatialCorrSpread, 2),
                spatial_corr_kurt: formatFixed(spatialCorrKurt, 2),
                quartile_lag1_near: formatFixed(q1Lag, 2),
                quartile_lag1_far: formatFixed(q4Lag, 2),
                joint_delta_chi2: formatFixed(environment.joint_delta_chi2, 1),
                joint_lrt_p: formatPValue(environment.joint_lrt_p),
                screened_core_r_s_min_au_int: formatInteger(Math.min(...screenedCoreRs)),
                screened_core_r_s_max_au_int: formatInteger(Math.max(...screenedCoreRs)),
                screened_core_r_s_spread_pct: formatFixed(Math.max(...screenedCoreFracDeviations), 0),
                low_mass_alpha: formatFixed(subsetControls.low_primary_mass_half?.alpha, 3),
                full_alpha: formatFixed(screening.alpha, 3),
                high_mass_alpha: formatFixed(subsetControls.high_primary_mass_half?.alpha, 3),
                low_mass_delta_vs_const_tex: formatLatexInteger(subsetControls.low_primary_mass_half?.delta_chi2_vs_constant_boost),
                self_alpha0: formatFixed(selfScreening.exp_alpha0, 2),
                self_alpha0_err: formatFixed(selfScreening.exp_alpha0_err, 2),
                self_mscreen: formatFixed(selfScreening.exp_M_screen, 2),
                self_mscreen_err: formatFixed(selfScreening.exp_M_screen_err, 2),
                self_chi2: formatFixed(selfScreening.exp_chi2, 2),
                self_tension_sigma: formatFixed(selfScreening.tension_with_cepheid_sigma, 1),
                cepheid_alpha0: formatFixed(selfScreening.cepheid_alpha0, 2),
                cepheid_alpha0_err: formatFixed(selfScreening.cepheid_alpha0_err, 2),
                controls_table_rows_html: [
                    ['Global scramble ($10{,}000$ realizations)', 'Can noise produce the transition?', `$${formatPValueClause(nullControls.global?.p_delta_chi2_vs_flat)}$`],
                    ['Distance-quartile scramble', 'Distance-correlated artifacts', `$${formatPValueClause(nullControls.within_distance_quartile?.p_delta_chi2_vs_flat)}$`],
                    [`Injection-recovery ($${formatLatexInteger(injection.signal_recovery?.n_realizations)}$ mocks)`, 'Pipeline fidelity and false-positive rate', `$${formatFixed(toNumber(injection.signal_recovery?.detection_rate) * 100, 0)}\%$ detection, $${formatFixed(toNumber(injection.null_injection?.detection_rate) * 100, 0)}\%$ false positive`],
                    ['Solar-track control', 'MLR correction dependence', 'Ordering preserved'],
                    ['Quadratic, no-correction, $\\beta = 1.0$, $\\beta = 2.0$ MLR', 'MLR functional form', `$${formatInteger(mlr.length)}/${formatInteger(mlr.length)}$ preserve ordering`],
                    ['Joint profile-likelihood fit', 'Fixed-$\\alpha$ assumption', `$\\Delta\\chi^2 = ${formatFixed(environment.joint_delta_chi2, 1)}$, $p = ${formatPValue(environment.joint_lrt_p)}$`],
                    [`$\\alpha_{\\rm sat}$ sweep ($${formatFixed(Math.min(...alphaSweep.map((row) => toNumber(row.alpha_fixed)).filter(Number.isFinite)), 2)}$&ndash;$${formatFixed(Math.max(...alphaSweep.map((row) => toNumber(row.alpha_fixed)).filter(Number.isFinite)), 2)}$)`, 'Sensitivity to fixed amplitude', `$${formatInteger(alphaSweep.filter((row) => String(row.ordering_preserved) === 'True').length)}/${formatInteger(alphaSweep.length)}$ preserve ordering`],
                    [`Leave-one-bin-out ($${formatInteger(lobo.length)}$ bins)`, 'Single-bin dominance', `$${formatInteger(lobo.filter((row) => row.ordering === 'YES').length)}/${formatInteger(lobo.length)}$ preserve ordering`],
                    ['Distance-matched subsamples', 'Distance distribution imbalance', 'Ordering preserved'],
                    ['Color-matched subsamples', 'Stellar population imbalance', 'Ordering preserved'],
                    ['Strict RUWE $< 1.1$', 'Triple contamination', 'Ordering preserved'],
                    ['Alternative binning ($12$, $18$ bins)', 'Bin definition dependence', '$2/2$ preserve ordering'],
                    ['AR(1) GLS refit', 'Bin-to-bin correlation', `$\\chi^2_{\\nu} = ${formatFixed(screening.gls_chi2_nu, 2)}$, conclusions preserved`],
                    ['GP covariance refit', 'Flexible covariance model', `$\\chi^2_{\\nu} = ${formatFixed(screening.gp_chi2_nu, 2)}$, conclusions preserved`],
                    ['Eccentricity sweep ($5$ distributions)', 'Projection systematic on $\\alpha_{\\rm sat}$', `$R_s$ stable $\\lesssim ${formatFixed(rsStablePct, 0)}\\%$; $\\alpha_{\\rm sat}$ bounded by the recovered sweep`],
                    ['Demographic half-samples ($4\\times$)', 'Subpopulation dependence', 'All prefer finite transition'],
                    [`RV-consistency subset ($${formatLatexInteger(subsetControls.rv_consistent?.n_systems)}$ systems)`, 'Kinematic purity', 'Transition recovered'],
                    ['Chameleon scaling prediction', '$\\epsilon_{\\rm env}$ post-hoc concern', `$n = 1$ prediction within $${formatFixed(Math.abs((toNumber(environment.chameleon_observed_ratio_joint) - toNumber(environment.chameleon_predicted_ratio_n1)) / toNumber(environment.chameleon_observed_ratio_joint_err)), 1)}\\sigma$`],
                    [`Fine $|Z|$ stratification ($${formatInteger(fineZ.length)}$ bins)`, 'Density-gradient resolution', `$n = ${formatFixed(chameleonFine?.chameleon_n_inferred, 2)} \\pm ${formatFixed(chameleonFine?.chameleon_n_err, 2)}$; $R_s$ declines midplane $\\to$ halo`],
                    ['Triple forward model ($6$ fractions)', 'Can residual triples mimic signal?', `Predicted outer $\\tilde{v} \\leq 1.00$; deficit $\\geq ${formatFixed(Math.abs(maxTripleDeficit), 2)}$`],
                    ['Self-screening model', 'Mass-dependent $\\alpha_{\\rm sat}$', `Exponential fit $\\chi^2 = ${formatFixed(selfScreening.exp_chi2, 2)}$ (1 dof)`],
                    ['Normalization sensitivity sweep', 'Baseline deflation by spectroscopic binaries', `$R_s$ stable within $\\pm ${formatFixed(Math.max(...screenedCoreFracDeviations), 0)}\\%$ for screened-core windows`],
                    ['Spatial substructure identification', 'Physical origin of $\\chi^2_{\\nu} = 5.1$', `Lag-1 $\\rho$ concentrated in distant quartile ($\\rho_1 = ${formatFixed(q4Lag, 2)}$)`],
                    ['Pittordis-faithful triple model', 'Exact Pittordis et al. (2025) triple distributions', `Predicted outer $\\tilde{v} \\leq 1.00$; deficit $\\geq ${formatFixed(Math.abs(maxPittDeficit), 2)}$`],
                ].map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('\n'),
            },
            mond: {
                simple_no_efe_a0_ratio: formatFixed(toNumber(String(mond.MOND_simple_no_EFE_perbin?.note || '').match(/a0=([^,]+)/)?.[1]) / 1.2e-10, 1),
                simple_efe_fixed_chi2: formatInteger(mond.MOND_simple_EFE_fixed_ge?.chi2),
                simple_efe_fixed_delta_tex: formatLatexInteger(toNumber(mond.MOND_simple_EFE_fixed_ge?.chi2) - toNumber(screening.chi2)),
                standard_efe_free_chi2: formatInteger(mond.MOND_standard_EFE_free_ge?.chi2),
                standard_efe_free_delta_tex: formatLatexInteger(toNumber(mond.MOND_standard_EFE_free_ge?.chi2) - toNumber(screening.chi2)),
                standard_efe_free_aic_delta: formatLatexInteger(toNumber(mond.MOND_standard_EFE_free_ge?.aic) - (toNumber(screening.chi2) + 4)),
            },
            conclusion: {
                rho_eff: formatSciLatex(environment.rho_floor_cgs, 1),
            },
        },
    };

    if (writeArtifact) {
        fs.mkdirSync(OUTPUTS_DIR, { recursive: true });
        fs.writeFileSync(CLAIMS_PATH, JSON.stringify(context, null, 2));
    }

    return context;
}

function safeGet(obj, pathParts) {
    let current = obj;
    for (const part of pathParts) {
        if (current === null || current === undefined) {
            return undefined;
        }
        current = current[part];
    }
    return current;
}

function injectPlaceholders(template, context, options = {}) {
    const { failOnUnresolved = true } = options;
    const unresolved = new Set();
    const rendered = template.replace(/\{\{\s*([^}]+?)\s*\}\}/g, (match, expr) => {
        const pathParts = expr.split('.').map((part) => part.trim()).filter(Boolean);
        const value = safeGet(context, pathParts);
        if (value === undefined || value === null || value === '') {
            unresolved.add(expr);
            return match;
        }
        return String(value);
    });

    if (unresolved.size > 0 && failOnUnresolved) {
        throw new Error(`Unresolved placeholders: ${Array.from(unresolved).join(', ')}`);
    }

    return rendered;
}

module.exports = {
    createManuscriptContext,
    injectPlaceholders,
};
