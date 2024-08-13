use std::ops::Deref;
pub mod structs;

use anndata_memory::IMAnnData;
use polars::{prelude::NamedFromOwned, series::Series};
use structs::StatisticsContainer;

use crate::shared::{statistics::number::whole, Direction};

pub fn compute_number(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    whole(data, direction)
}

pub fn compute_sum(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::sum::whole(data, direction)
}

pub fn compute_variance(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::variance::whole(data, direction)
}

pub fn compute_min_max(
    adata: &IMAnnData,
    direction: Direction,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::minmax::whole(data, direction)
}

pub fn compute_std_dev(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::stddev::whole(data, direction)
}

pub fn compute_qc_variables(adata: &IMAnnData) -> anyhow::Result<StatisticsContainer> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();

    let n_per_gene = crate::shared::statistics::number::whole(data, Direction::Column)?;
    let n_per_cell = crate::shared::statistics::number::whole(data, Direction::Row)?;
    let sum_per_gene = crate::shared::statistics::sum::whole(data, Direction::Column)?;
    let sum_per_cell = crate::shared::statistics::sum::whole(data, Direction::Row)?;
    let var_per_gene = crate::shared::statistics::variance::whole(data, Direction::Column)?;
    let var_per_cell = crate::shared::statistics::variance::whole(data, Direction::Row)?;
    let std_dev_per_gene = crate::shared::statistics::stddev::whole(data, Direction::Column)?;
    let std_dev_per_cell = crate::shared::statistics::stddev::whole(data, Direction::Row)?;

    Ok(StatisticsContainer {
        num_per_cell: n_per_cell,
        num_per_gene: n_per_gene,
        expr_per_gene: sum_per_gene,
        expr_per_cell: sum_per_cell,
        variance_per_gene: var_per_gene,
        variance_per_cell: var_per_cell,
        std_dev_per_cell,
        std_dev_per_gene,
    })
}

pub fn qc_vars_inplace(adata: &IMAnnData) -> anyhow::Result<()> {
    let data = compute_qc_variables(adata)?;

    let mut obs_df = adata.obs().get_data();
    let mut var_df = adata.var().get_data();

    let num_cell_series = Series::from_vec("num_genes_per_cell", data.num_per_cell);
    let num_genes_series = Series::from_vec("num_cells_per_gene", data.num_per_gene);
    let expr_per_gene_series = Series::from_vec("sum_expr_per_gene", data.expr_per_gene);
    let expr_per_cell_series = Series::from_vec("sum_expr_per_cell", data.expr_per_cell);
    let var_per_gene_series = Series::from_vec("var_expr_per_gene", data.variance_per_gene);
    let var_per_cell_series = Series::from_vec("var_expr_per_cell", data.variance_per_cell);
    let std_dev_per_gene_series = Series::from_vec("std_dev_per_gene", data.std_dev_per_gene);
    let std_dev_per_cell_series = Series::from_vec("std_dev_per_cell", data.std_dev_per_cell);

    obs_df.with_column(num_cell_series)?;
    obs_df.with_column(expr_per_cell_series)?;
    obs_df.with_column(var_per_cell_series)?;
    obs_df.with_column(std_dev_per_cell_series)?;

    var_df.with_column(num_genes_series)?;
    var_df.with_column(expr_per_gene_series)?;
    var_df.with_column(var_per_gene_series)?;
    var_df.with_column(std_dev_per_gene_series)?;

    adata.obs().set_data(obs_df)?;
    adata.var().set_data(var_df)?;

    Ok(())
}
