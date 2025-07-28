use anndata_memory::{IMAnnData, IMArrayElement};
use num_traits::Float;
use polars::{frame::DataFrame, prelude::Column};
use single_utilities::types::Direction;

use crate::{shared::statistics::ComputeTopSegmentProportions, ComputeNonZero, ComputeSum};

fn describe_obs(
    adata: &IMAnnData,
    x: &IMArrayElement,
    expr_type: &str,
    var_type: &str,
    qc_vars: &[&str],
    percent_top: &[usize],
    log1p: bool,
) -> anyhow::Result<DataFrame> {
    let n_obs = adata.n_obs();
    let n_vars = adata.n_vars();

    let n_genes_by_counts: Vec<u32> = x.nonzero_whole(&Direction::ROW)?;
    let total_counts: Vec<f64> = x.sum_whole(&Direction::ROW)?;

    let mut columns = vec![];

    let col_name = format!("n_{}_by_{}", var_type, expr_type);
    columns.push(Column::new(col_name.into(), n_genes_by_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = n_genes_by_counts
            .iter()
            .map(|&x| (x as f64 + 1.0).ln())
            .collect();
        let col_name = format!("log1p_n_{}_by_{}", var_type, expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    let col_name = format!("total_{}", expr_type);
    columns.push(Column::new(col_name.into(), total_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = total_counts.iter().map(|&x| (x + 1.0).ln()).collect();
        let col_name = format!("log1p_total_{}", expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    if !percent_top.is_empty() {
        use crate::shared::statistics::ComputeTopSegmentProportions;
        let proportions = x.top_segment_proportions(&Direction::ROW, percent_top)?;
        for (i, &n) in percent_top.iter().enumerate() {
            let values: Vec<f64> = proportions.column(i).iter().map(|&x| x * 100.0).collect();
            let col_name = format!("pct_{}_in_top_{}_{}", expr_type, n, var_type);
            columns.push(Column::new(col_name.into(), values));
        }
    }

    for qc_var in qc_vars {
        let var_mask = adata
            .var()
            .get_column_from_df(qc_var)?
            .bool()?
            .into_iter()
            .map(|x| x.unwrap_or(false))
            .collect::<Vec<bool>>();

        let qc_var_indices: Vec<usize> = var_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &mask)| if mask { Some(i) } else { None })
            .collect();

        let qc_totals = if !qc_var_indices.is_empty() {
            let mut mask = vec![false; n_vars];
            for &idx in &qc_var_indices {
                if idx < n_vars {
                    mask[idx] = true;
                }
            }

            x.sum_whole_masked(&Direction::ROW, &mask)?
        } else {
            vec![0.0; n_obs]
        };

        let col_name = format!("total_{}_{}", expr_type, qc_var);
        columns.push(Column::new(col_name.into(), qc_totals.clone()));

        if log1p {
            let log_values: Vec<f64> = qc_totals.iter().map(|&x| (x + 1.0).ln()).collect();
            let col_name = format!("log1p_total_{}_{}", expr_type, qc_var);
            columns.push(Column::new(col_name.into(), log_values));
        }

        let pct_values: Vec<f64> = qc_totals
            .iter()
            .zip(total_counts.iter())
            .map(|(&qc, &total)| if total > 0.0 { qc / total * 100.0 } else { 0.0 })
            .collect();
        let col_name = format!("pct_{}_{}", expr_type, qc_var);
        columns.push(Column::new(col_name.into(), pct_values));
    }

    DataFrame::new(columns).map_err(Into::into)
}

fn describe_var(
    adata: &IMAnnData,
    x: &anndata_memory::IMArrayElement,
    expr_type: &str,
    var_type: &str,
    log1p: bool,
) -> anyhow::Result<DataFrame> {
    let n_obs = adata.n_obs();

    let n_cells_by_counts: Vec<u32> = x.nonzero_whole(&Direction::COLUMN)?;
    let total_counts: Vec<f64> = x.sum_whole(&Direction::COLUMN)?;

    let mean_counts: Vec<f64> = total_counts
        .iter()
        .map(|&total| total / n_obs as f64)
        .collect();

    let mut columns = vec![];

    let col_name = format!("n_cells_by_{}", expr_type);
    columns.push(Column::new(col_name.into(), n_cells_by_counts.clone()));

    let col_name = format!("mean_{}", expr_type);
    columns.push(Column::new(col_name.into(), mean_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = mean_counts.iter().map(|&x| (x + 1.0).ln()).collect();
        let col_name = format!("log1p_mean_{}", expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    let pct_dropout: Vec<f64> = n_cells_by_counts
        .iter()
        .map(|&n| (1.0 - n as f64 / n_obs as f64) * 100.0)
        .collect();
    let col_name = format!("pct_dropout_by_{}", expr_type);
    columns.push(Column::new(col_name.into(), pct_dropout));

    let col_name = format!("total_{}", expr_type);
    columns.push(Column::new(col_name.into(), total_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = total_counts.iter().map(|&x| (x + 1.0).ln()).collect();
        let col_name = format!("log1p_total_{}", expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    DataFrame::new(columns).map_err(Into::into)
}

pub fn calculate_qc_metrics(
    adata: &IMAnnData,
    expr_type: Option<&str>,
    var_type: Option<&str>,
    qc_vars: Option<Vec<&str>>,
    percent_top: Option<Vec<usize>>,
    layer: Option<&str>,
    use_raw: bool,
    inplace: bool,
    log1p: bool,
) -> anyhow::Result<Option<(DataFrame, DataFrame)>> {
    let expr_type = expr_type.unwrap_or("counts");
    let var_type = var_type.unwrap_or("genes");
    let qc_vars = qc_vars.unwrap_or_default();
    let percent_top = percent_top.unwrap_or_else(|| vec![50, 100, 200, 500]);

    let x = if let Some(layer_name) = layer {
        adata.layers().get_array_shallow(layer_name)?
    } else if use_raw {
        return Err(anyhow::anyhow!("Raw data access not yet implemented"));
    } else {
        adata.x()
    };

    let obs_metrics = describe_obs(
        adata,
        &x,
        expr_type,
        var_type,
        &qc_vars,
        &percent_top,
        log1p,
    )?;

    let var_metrics = describe_var(adata, &x, expr_type, var_type, log1p)?;

    if inplace {
        let mut obs_df = adata.obs().get_data();
        for col in obs_metrics.get_columns() {
            obs_df.with_column(col.clone())?;
        }
        adata.obs().set_data(obs_df)?;

        let mut var_df = adata.var().get_data();
        for col in var_metrics.get_columns() {
            var_df.with_column(col.clone())?;
        }
        adata.var().set_data(var_df)?;

        Ok(None)
    } else {
        Ok(Some((obs_metrics, var_metrics)))
    }
}

pub fn qc_metrics(adata: &IMAnnData) -> anyhow::Result<()> {
    let var_names = adata.var_names();
    let mito_mask: Vec<bool> = var_names
        .iter()
        .map(|name| name.starts_with("MT-") || name.starts_with("mt-"))
        .collect();

    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("mito".into(), mito_mask))?;
    adata.var().set_data(var_df)?;

    calculate_qc_metrics(
        adata,
        Some("counts"),
        Some("genes"),
        Some(vec!["mito"]),
        Some(vec![50, 100, 200, 500]),
        None,
        false,
        true,
        true,
    )?;

    Ok(())
}
