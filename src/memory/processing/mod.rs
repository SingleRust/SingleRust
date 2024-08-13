use anndata::data::SelectInfoElem;
use anndata_memory::IMAnnData;
use ndarray::{Array1, Axis};
use noisy_float::types::n64;

use crate::{shared::FlexValue, Direction};
use ndarray_stats::QuantileExt;


fn calculate_cell_stats(
    adata: &IMAnnData,
    need_gene_count: bool,
) -> anyhow::Result<(Option<Vec<u32>>, Vec<f64>)> {
    let n_genes_per_cell = if need_gene_count {
        Some(crate::memory::statistics::compute_number(adata, Direction::Row)?)
    } else {
        None
    };
    let sum_genes_per_cell = crate::memory::statistics::compute_sum(adata, Direction::Row)?;
    Ok((n_genes_per_cell, sum_genes_per_cell))
}

fn create_filter_mask(
    n_obs: usize,
    n_genes_per_cell: &Option<Vec<u32>>,
    sum_genes_per_cell: &[f64],
    lower_lim: &FlexValue,
    upper_lim: &FlexValue,
    lower_percentile: f64,
    upper_percentile: f64,
) -> Array1<bool> {
    Array1::from_vec((0..n_obs).map(|i| {
        match (lower_lim, upper_lim) {
            (FlexValue::Absolute(lower), FlexValue::Absolute(upper)) => {
                let n_genes = n_genes_per_cell.as_ref().unwrap()[i];
                n_genes >= *lower && n_genes <= *upper
            }
            (FlexValue::Relative(_), FlexValue::Relative(_)) => {
                let sum_genes = sum_genes_per_cell[i];
                sum_genes >= lower_percentile && sum_genes <= upper_percentile
            }
            (FlexValue::Absolute(lower), FlexValue::Relative(_)) => {
                let n_genes = n_genes_per_cell.as_ref().unwrap()[i];
                let sum_genes = sum_genes_per_cell[i];
                n_genes >= *lower && sum_genes <= upper_percentile
            }
            (FlexValue::Relative(_), FlexValue::Absolute(upper)) => {
                let n_genes = n_genes_per_cell.as_ref().unwrap()[i];
                let sum_genes = sum_genes_per_cell[i];
                sum_genes >= lower_percentile && n_genes <= *upper
            }
            _ => true,
        }
    }).collect())
}

pub fn filter_cells_inplace(
    adata: &mut IMAnnData,
    lower_lim: FlexValue,
    upper_lim: FlexValue,
) -> anyhow::Result<()> {
    let need_gene_count = matches!(lower_lim, FlexValue::Absolute(_)) || matches!(upper_lim, FlexValue::Absolute(_));
    let (n_genes_per_cell, sum_genes_per_cell) = calculate_cell_stats(adata, need_gene_count)?;

    let (lower_percentile, upper_percentile) = calculate_percentiles(&sum_genes_per_cell, &lower_lim, &upper_lim)?;

    let mask = create_filter_mask(
        adata.n_obs(),
        &n_genes_per_cell,
        &sum_genes_per_cell,
        &lower_lim,
        &upper_lim,
        lower_percentile,
        upper_percentile,
    );

    let selection = crate::shared::processing::get_select_info_obs(Some(mask.view()))?;
    let selection_refs: Vec<&SelectInfoElem> = selection.iter().collect();

    adata.subset_inplace(selection_refs.as_slice())
}

pub fn filter_cells(
    adata: &IMAnnData,
    lower_lim: FlexValue,
    upper_lim: FlexValue,
) -> anyhow::Result<IMAnnData> {
    let need_gene_count = matches!(lower_lim, FlexValue::Absolute(_)) || matches!(upper_lim, FlexValue::Absolute(_));
    let (n_genes_per_cell, sum_genes_per_cell) = calculate_cell_stats(adata, need_gene_count)?;

    let (lower_percentile, upper_percentile) = calculate_percentiles(&sum_genes_per_cell, &lower_lim, &upper_lim)?;

    let mask = create_filter_mask(
        adata.n_obs(),
        &n_genes_per_cell,
        &sum_genes_per_cell,
        &lower_lim,
        &upper_lim,
        lower_percentile,
        upper_percentile,
    );

    let selection = crate::shared::processing::get_select_info_obs(Some(mask.view()))?;
    let selection_refs: Vec<&SelectInfoElem> = selection.iter().collect();

    adata.subset(selection_refs.as_slice())
}


fn calculate_percentiles(values: &[f64], lower_lim: &FlexValue, upper_lim: &FlexValue) -> anyhow::Result<(f64, f64)> {
    let mut arr = Array1::from_vec(values.iter().map(|&x| n64(x)).collect());

    let lower_percentile = match lower_lim {
        FlexValue::Relative(p) => arr.quantile_axis_mut(Axis(0), n64(*p), &ndarray_stats::interpolate::Linear)
            .map_err(|e| anyhow::anyhow!("Error calculating lower percentile: {}", e))?
            .into_scalar().raw(),
        _ => f64::MIN,
    };

    let upper_percentile = match upper_lim {
        FlexValue::Relative(p) => arr.quantile_axis_mut(Axis(0), n64(*p), &ndarray_stats::interpolate::Linear)
            .map_err(|e| anyhow::anyhow!("Error calculating upper percentile: {}", e))?
            .into_scalar().raw(),
        _ => f64::MAX,
    };

    Ok((lower_percentile, upper_percentile))
}