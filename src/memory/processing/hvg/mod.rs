use crate::shared::processing::{calculate_dispersion_stats, fit_svr, get_mean_bins, standardize_log_form_vec, FlavorType, HVGParams};
use crate::{ComputeSum, ComputeVariance};
use anndata_memory::{IMAnnData, IMArrayElement};
use nalgebra::min;
use polars::prelude::Column;
use single_algebra::Direction;

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
    for i in 0..bin_means.len() {
        if bin_stds[i].is_nan() {
            bin_stds[i] = bin_means[i];
            bin_means[i] = 0.0;
        }
    }

    Ok(())
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
        let mean = bin_means[bin_idx];
        let std = bin_stds[bin_idx];

        if !std.is_nan() && std > 0.0 {
            normalized_dispersions[i] = (disp - mean) / std;
        } else {
            normalized_dispersions[i] = 0.0;
        }
    }

    Ok(normalized_dispersions)
}

fn subset_genes(
    means: &[f64],
    dispersion_norm: &[f64],
    n_top_genes: Option<usize>,
    min_mean: f64,
    max_mean: f64,
    min_dispersion: f64
) -> anyhow::Result<Vec<bool>> {

    let mut highly_variable = vec![false; means.len()];

    let valid_by_mean: Vec<bool> = means.iter()
        .map(|&mean| mean >= min_mean && mean <= max_mean)
        .collect();

    let clear_dispersions: Vec<f64> = dispersion_norm.iter()
        .map(|&d| if d.is_nan() {f64::NEG_INFINITY} else {d})
        .collect();

    if let Some(n_top) = n_top_genes {
        let valid_n_top = min(n_top, means.len());

        let mut valid_dispersions: Vec<f64> = clear_dispersions.iter()
            .enumerate()
            .filter(|(i, _)| valid_by_mean[*i])
            .map(|(_, &d)| d)
            .filter(|&d| !d.is_nan())
            .collect();

        if valid_n_top > valid_dispersions.len() {
            for i in 0..means.len() {
                if valid_by_mean[i] && !clear_dispersions[i].is_nan() {
                    highly_variable[i] = true;
                }
            }
        } else {
            valid_dispersions.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let cutoff = valid_dispersions[valid_n_top - 1];

            for i in 0..means.len() {
                highly_variable[i] = valid_by_mean[i] && clear_dispersions[i] >= cutoff;
            }
        }
    } else {
        for i in 0..means.len() {
            highly_variable[i] = valid_by_mean[i] && clear_dispersions[i] > min_dispersion;
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
    let means: Vec<f64> = x
        .sum_whole(&Direction::COLUMN)?
        .iter()
        .map(|sum: &f64| sum / n_obs as f64)
        .collect();

    let variances: Vec<f64> = x.variance_whole::<u32, f64>(&Direction::COLUMN)?;

    let dispersions: Vec<f64> = means
        .iter()
        .zip(variances.iter())
        .map(|(&mean, &var)| if mean > 0.0 { var / mean } else { var / 1e-12 }) // handle similar to scanpy
        .collect();
    let log_means: Vec<f64> = means.iter().map(|&x| (x+1.0).ln()).collect();
    let log_dispersions: Vec<f64> = dispersions.iter().map(|&x| x.ln()).collect();

    let n_bins = params.n_bins;
    let (mean_bins, bin_indices) = get_mean_bins(&log_means, n_bins)?;

    let (mut bin_means, mut bin_stds) = calculate_dispersion_stats(&log_dispersions, &bin_indices, &mean_bins)?;

    postprocess_seurat_dispersions(&mut bin_means, &mut bin_stds)?;

    let normalized_dispersions = normalize_dispersions(&log_dispersions, &bin_indices, &bin_means, &bin_stds)?;

    let standardized_dispersions = standardize_log_form_vec(&normalized_dispersions);

    let highly_variable = subset_genes(
        &means,
        &standardized_dispersions,
        params.n_top_genes,
        params.min_mean,
        params.max_mean,
        params.min_dispersion,
    )?;

    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("means".into(), log_means))?;
    var_df.with_column(Column::new("dispersions".into(), log_dispersions))?;
    var_df.with_column(Column::new(
        "dispersions_norm".into(),
        normalized_dispersions,
    ))?;
    var_df.with_column(Column::new("highly_variable".into(), highly_variable))?;
    var_df.with_column(Column::new(
        "dispersions_normalized_standardized".into(),
        standardized_dispersions,
    ))?;

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
