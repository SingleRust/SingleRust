use crate::shared::processing::{fit_svr, standardize_log_form_vec, FlavorType, HVGParams};
use crate::{ComputeSum, ComputeVariance};
use anndata_memory::{IMAnnData, IMArrayElement};
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
        .map(|(&mean, &var)| if mean > 0.0 { var / mean } else { 0.0 })
        .collect();
    let log_means: Vec<f64> = means.iter().map(|&x| x.ln()).collect();
    let log_dispersions: Vec<f64> = dispersions.iter().map(|&x| x.ln()).collect();

    let mut normalized_dispersions = if let Some(batch_key) = params.batch_key {
        let batch_col = adata.obs().get_column_from_df(&batch_key)?;
        crate::shared::processing::normalize_by_batch(
            &log_means,
            &log_dispersions,
            &batch_col,
            params.n_bins,
        )?
    } else {
        crate::shared::processing::normalize_per_bin(
            &log_dispersions,
            &log_dispersions,
            params.n_bins,
        )?
    };

    normalized_dispersions = normalized_dispersions
        .into_iter()
        .map(|x| x.max(params.min_dispersion).min(params.max_dispersion))
        .collect();

    let standardized_dispersions = standardize_log_form_vec(&normalized_dispersions);

    let mut highly_variable = vec![false; means.len()];
    if let Some(n_top) = params.n_top_genes {
        let mut dispersion_indices: Vec<usize> = (0..standardized_dispersions.len()).collect();
        dispersion_indices.sort_by(|&a, &b| {
            standardized_dispersions[b]
                .partial_cmp(&standardized_dispersions[a])
                .unwrap()
        });

        for &idx in dispersion_indices.iter().take(n_top) {
            if means[idx] >= params.min_mean && means[idx] <= params.max_mean {
                highly_variable[idx] = true;
            }
        }
    } else {
        for i in 0..means.len() {
            highly_variable[i] = means[i] >= params.min_mean
                && means[i] <= params.max_mean
                && standardized_dispersions[i] >= params.min_dispersion
                && standardized_dispersions[i] <= params.max_dispersion;
        }
    }

    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("means".into(), means))?;
    var_df.with_column(Column::new("dispersions".into(), dispersions))?;
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
