use crate::shared::processing::{fit_svr, standardize_log_form_vec, FlavorType, HVGParams};
use crate::{ComputeSum, ComputeVariance};
use anndata_memory::{IMAnnData, IMArrayElement};
use polars::prelude::Column;
use single_utilities::types::Direction;

pub fn compute_highly_variable_genes(
    adata: &IMAnnData,
    params: Option<HVGParams>,
) -> anyhow::Result<()> {
    let params = params.unwrap_or_default();
    let x = adata.x();

    match params.flavor {
        FlavorType::Seurat => compute_seurat_hvg(adata, &x, params),
        FlavorType::CellRanger => compute_cell_ranger_hvg(adata, &x, params),
        FlavorType::SVR => compute_svr_hvg(adata, &x, params),
    }
}

fn postprocess_seurat_dispersions(
    bin_means: &mut [f64],
    bin_stds: &mut [f64],
) -> anyhow::Result<()> {
    // This matches Python's _postprocess_dispersions_seurat
    for i in 0..bin_means.len() {
        if bin_stds[i].is_nan() {
            // For single-gene bins, set std = mean and mean = 0
            // This effectively sets normalized dispersion to 1
            bin_stds[i] = bin_means[i];
            bin_means[i] = 0.0;
        }
    }
    Ok(())
}

fn equal_width_binning(log_means: &[f64], n_bins: usize) -> anyhow::Result<(Vec<usize>, Vec<f64>)> {
    // Find min and max (excluding NaN)
    let mut valid_means: Vec<f64> = log_means
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();

    if valid_means.is_empty() {
        return Err(anyhow::anyhow!("No valid mean values found"));
    }

    valid_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_mean = valid_means[0];
    let max_mean = valid_means[valid_means.len() - 1];

    // Create equal-width bins
    let bin_width = (max_mean - min_mean) / n_bins as f64;
    let mut bin_edges = vec![0.0; n_bins + 1];

    for (i, edge) in bin_edges.iter_mut().enumerate().take(n_bins + 1) {
        *edge = min_mean + (i as f64) * bin_width;
    }

    // Make sure the last edge includes the maximum value
    bin_edges[n_bins] = max_mean + 1e-10;

    // Assign each gene to a bin
    let mut bin_indices = vec![0; log_means.len()];

    for (i, &mean) in log_means.iter().enumerate() {
        if !mean.is_finite() {
            bin_indices[i] = 0; // Assign NaN/inf to first bin
            continue;
        }

        // Find which bin this value belongs to
        let mut bin_idx = 0;
        for j in 0..n_bins {
            if mean >= bin_edges[j] && mean < bin_edges[j + 1] {
                bin_idx = j;
                break;
            }
        }

        // Handle edge case where mean == max_mean
        if mean == max_mean {
            bin_idx = n_bins - 1;
        }

        bin_indices[i] = bin_idx;
    }

    Ok((bin_indices, bin_edges))
}

fn calculate_bin_stats(
    log_dispersions: &[f64],
    bin_indices: &[usize],
    n_bins: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let mut bin_values: Vec<Vec<f64>> = vec![Vec::new(); n_bins];

    // Collect values for each bin (excluding NaN)
    for (i, &bin_idx) in bin_indices.iter().enumerate() {
        let disp = log_dispersions[i];
        if !disp.is_nan() && bin_idx < n_bins {
            bin_values[bin_idx].push(disp);
        }
    }

    let mut bin_means = vec![0.0; n_bins];
    let mut bin_stds = vec![0.0; n_bins];

    for bin_idx in 0..n_bins {
        let values = &bin_values[bin_idx];

        if values.is_empty() {
            bin_means[bin_idx] = f64::NAN;
            bin_stds[bin_idx] = f64::NAN;
        } else if values.len() == 1 {
            // Single gene in bin - Python sets std to NaN
            bin_means[bin_idx] = values[0];
            bin_stds[bin_idx] = f64::NAN;
        } else {
            // Calculate mean
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            bin_means[bin_idx] = mean;

            // Calculate standard deviation
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

            bin_stds[bin_idx] = variance.sqrt();
        }
    }

    Ok((bin_means, bin_stds))
}

fn normalize_dispersions(
    log_dispersions: &[f64],
    bin_indices: &[usize],
    bin_means: &[f64],
    bin_stds: &[f64],
) -> anyhow::Result<Vec<f64>> {
    let mut normalized_dispersions = vec![0.0; log_dispersions.len()];

    for (i, &disp) in log_dispersions.iter().enumerate() {
        let bin_idx = bin_indices[i];

        if bin_idx >= bin_means.len() {
            normalized_dispersions[i] = f64::NAN;
            continue;
        }

        let mean = bin_means[bin_idx];
        let std = bin_stds[bin_idx];

        if disp.is_nan() || mean.is_nan() || std.is_nan() || std == 0.0 {
            normalized_dispersions[i] = f64::NAN;
        } else {
            normalized_dispersions[i] = (disp - mean) / std;
        }
    }

    Ok(normalized_dispersions)
}

fn subset_genes(
    log_means: &[f64], // These are already log-transformed
    dispersion_norm: &[f64],
    n_top_genes: Option<usize>,
    min_mean: f64,
    max_mean: f64,
    min_dispersion: f64,
) -> anyhow::Result<Vec<bool>> {
    let mut highly_variable = vec![false; log_means.len()];

    if let Some(n_top) = n_top_genes {
        // Python's approach for n_top_genes:
        // 1. First, remove NaN values to compute threshold
        let non_nan_dispersions: Vec<f64> = dispersion_norm
            .iter()
            .filter(|&&d| !d.is_nan())
            .copied()
            .collect();

        if non_nan_dispersions.is_empty() {
            return Ok(highly_variable);
        }

        // Find the nth highest value
        let n_to_select = n_top.min(non_nan_dispersions.len());
        let mut sorted_dispersions = non_nan_dispersions.clone();
        sorted_dispersions.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = sorted_dispersions[n_to_select - 1];

        // 2. Now apply threshold to nan_to_num version (NaN → -inf)
        for i in 0..dispersion_norm.len() {
            let disp_value = if dispersion_norm[i].is_nan() {
                f64::NEG_INFINITY // Python uses -inf for NaN in final selection
            } else {
                dispersion_norm[i]
            };

            if disp_value >= threshold {
                highly_variable[i] = true;
            }
        }
    } else {
        // Original cutoff-based selection
        // Python applies nan_to_num (NaN → 0) before checking bounds
        let clean_dispersions: Vec<f64> = dispersion_norm
            .iter()
            .map(|&d| if d.is_nan() { 0.0 } else { d })
            .collect();

        // Apply mean filters
        let valid_by_mean: Vec<bool> = log_means
            .iter()
            .map(|&log_mean| log_mean > min_mean && log_mean < max_mean)
            .collect();

        for i in 0..log_means.len() {
            highly_variable[i] = valid_by_mean[i] && clean_dispersions[i] > min_dispersion;
        }
    }

    Ok(highly_variable)
}

fn compute_seurat_hvg(
    adata: &IMAnnData,
    x: &IMArrayElement,
    params: HVGParams,
) -> anyhow::Result<()> {
    let n_obs = adata.n_obs();

    // Calculate means from raw counts
    let raw_means: Vec<f64> = x
        .sum_whole(&Direction::COLUMN)?
        .iter()
        .map(|sum: &f64| sum / n_obs as f64)
        .collect();

    let variances: Vec<f64> = x.variance_whole::<u32, f64>(&Direction::COLUMN)?;

    // Calculate dispersions with proper handling of zero means
    let dispersions: Vec<f64> = raw_means
        .iter()
        .zip(variances.iter())
        .map(|(&mean, &var)| {
            let safe_mean = if mean > 1e-12 { mean } else { 1e-12 };
            var / safe_mean
        })
        .collect();

    // For Seurat flavor, use log1p of means for binning and storage
    // This matches what Python does AFTER reverting log normalization
    let log1p_means: Vec<f64> = raw_means.iter().map(|&x| (x + 1.0).ln()).collect();

    // Log dispersions with NaN for zero dispersions (matching Python)
    let log_dispersions: Vec<f64> = dispersions
        .iter()
        .map(|&x| {
            if x > 0.0 {
                x.ln()
            } else {
                f64::NAN // Python sets dispersion[dispersion == 0] = np.nan
            }
        })
        .collect();

    let n_bins = params.n_bins;

    // Use equal-width binning on log1p_means (like Python's pd.cut)
    let (bin_indices, _) = equal_width_binning(&log1p_means, n_bins)?;

    // Calculate mean and std for each bin
    let (mut bin_means, mut bin_stds) =
        calculate_bin_stats(&log_dispersions, &bin_indices, n_bins)?;

    // Handle single-gene bins (like Python's _postprocess_dispersions_seurat)
    postprocess_seurat_dispersions(&mut bin_means, &mut bin_stds)?;

    // Normalize dispersions
    let normalized_dispersions =
        normalize_dispersions(&log_dispersions, &bin_indices, &bin_means, &bin_stds)?;

    // Select highly variable genes using raw means for filtering
    let highly_variable = subset_genes(
        &log1p_means, // Pass log-transformed means
        &normalized_dispersions,
        params.n_top_genes,
        params.min_mean,
        params.max_mean,
        params.min_dispersion,
    )?;

    // Store results - IMPORTANT: Store log1p means to match Python
    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("means".into(), log1p_means))?; // Store log1p means like Python
    var_df.with_column(Column::new("dispersions".into(), log_dispersions))?; // Store log dispersions
    var_df.with_column(Column::new(
        "dispersions_norm".into(),
        normalized_dispersions,
    ))?;
    var_df.with_column(Column::new("highly_variable".into(), highly_variable))?;

    adata.var().set_data(var_df)
}

fn compute_cell_ranger_hvg(
    _adata: &IMAnnData,
    _x: &IMArrayElement,
    _params: HVGParams,
) -> anyhow::Result<()> {
    todo!("Cell Ranger flavor is not implemented yet!")
}

fn compute_svr_hvg(adata: &IMAnnData, x: &IMArrayElement, params: HVGParams) -> anyhow::Result<()> {
    let n_obs = adata.n_obs();
    let means: Vec<f64> = x
        .sum_whole(&Direction::COLUMN)?
        .iter()
        .map(|sum: &f64| sum / n_obs as f64)
        .collect();

    let variances: Vec<f64> = x.variance_whole::<u32, f64>(&Direction::COLUMN)?;

    let log_means: Vec<f64> = means.iter().map(|&x| x.ln()).collect();
    let log_variances: Vec<f64> = variances.iter().map(|&x| x.ln()).collect();

    let (residuals, y_pred) = fit_svr(&log_means, &log_variances)?;
    let standardized_results = standardize_log_form_vec(&residuals);

    let mut highly_variable = vec![false; means.len()];
    if let Some(n_top) = params.n_top_genes {
        let mut indices: Vec<usize> = (0..standardized_results.len()).collect();
        indices.sort_by(|&a, &b| {
            standardized_results[b]
                .partial_cmp(&standardized_results[a])
                .unwrap()
        });
        for &idx in indices.iter().take(n_top) {
            if means[idx] >= params.min_mean && means[idx] <= params.max_mean {
                highly_variable[idx] = true;
            }
        }
    } else {
        for i in 0..means.len() {
            highly_variable[i] = means[i] >= params.min_mean
                && means[i] <= params.max_mean
                && standardized_results[i] > params.min_dispersion;
        }
    }

    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("means".into(), means))?;
    var_df.with_column(Column::new("variances".into(), variances))?;
    var_df.with_column(Column::new("residuals".into(), residuals))?;
    var_df.with_column(Column::new("highly_variable".into(), highly_variable))?;
    var_df.with_column(Column::new(
        "residuals_standardized".into(),
        standardized_results,
    ))?;
    var_df.with_column(Column::new("mean_variance_trend".into(), y_pred))?;

    adata.var().set_data(var_df)
}
