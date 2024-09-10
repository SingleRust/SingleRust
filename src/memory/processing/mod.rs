use anndata::data::SelectInfoElem;
use anndata_memory::DeepClone;
use anndata_memory::IMAnnData;
use anyhow::Ok;
use ndarray::{Array1, Axis};
use noisy_float::types::n64;

use crate::{shared::FlexValue, Direction};
use ndarray_stats::QuantileExt;
use log::{log, Level};
mod scale;
mod transform;

pub mod dim_red;

fn calculate_cell_stats(
    adata: &IMAnnData,
    need_gene_count: bool,
) -> anyhow::Result<(Option<Vec<u32>>, Vec<f64>)> {
    let n_genes_per_cell = if need_gene_count {
        Some(crate::memory::statistics::compute_number(
            adata,
            Direction::Row,
        )?)
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
    Array1::from_vec(
        (0..n_obs)
            .map(|i| match (lower_lim, upper_lim) {
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
                (FlexValue::Absolute(lower), FlexValue::None) => {
                    let n_genes = n_genes_per_cell.as_ref().unwrap()[i];
                    log!(Level::Debug, "Number of genes {}, limit: {}", n_genes, *lower);
                    n_genes >= *lower
                }
                (FlexValue::None, FlexValue::Absolute(upper)) => {
                    let n_genes = n_genes_per_cell.as_ref().unwrap()[i];
                    n_genes <= *upper
                }
                (FlexValue::Relative(_), FlexValue::None) => {
                    let sum_genes = sum_genes_per_cell[i];
                    sum_genes >= lower_percentile
                }
                (FlexValue::None, FlexValue::Relative(_)) => {
                    let sum_genes = sum_genes_per_cell[i];
                    sum_genes <= upper_percentile
                }
                (FlexValue::None, FlexValue::None) => true,
            })
            .collect(),
    )
}

// TODO: refactor this and optimize code, here are unnecessary calculations
pub fn filter_cells_inplace(
    adata: &mut IMAnnData,
    lower_lim: FlexValue,
    upper_lim: FlexValue,
) -> anyhow::Result<()> {
    let need_gene_count =
        matches!(lower_lim, FlexValue::Absolute(_)) || matches!(upper_lim, FlexValue::Absolute(_));
    log!(Level::Debug, "Performing filtering of all cells");    

    let (n_genes_per_cell, sum_genes_per_cell) = calculate_cell_stats(adata, need_gene_count)?;

    log!(Level::Debug, "Calculating all sums");
    let (lower_percentile, upper_percentile) =
        calculate_percentiles(&sum_genes_per_cell, &lower_lim, &upper_lim)?;

    log!(Level::Debug, "Creating a filtering mask");
    let mask = create_filter_mask(
        adata.n_obs(),
        &n_genes_per_cell,
        &sum_genes_per_cell,
        &lower_lim,
        &upper_lim,
        lower_percentile,
        upper_percentile,
    );

    log!(Level::Debug, "Getting selection info");
    let selection = crate::shared::processing::get_select_info_obs(Some(mask.view()))?;
    let selection_refs: Vec<&SelectInfoElem> = selection.iter().collect();

    log!(Level::Debug, "Subsetting inplace");
    adata.subset_inplace(selection_refs.as_slice())
}

pub fn filter_cells(
    adata: &IMAnnData,
    lower_lim: FlexValue,
    upper_lim: FlexValue,
) -> anyhow::Result<IMAnnData> {
    let need_gene_count =
        matches!(lower_lim, FlexValue::Absolute(_)) || matches!(upper_lim, FlexValue::Absolute(_));
    let (n_genes_per_cell, sum_genes_per_cell) = calculate_cell_stats(adata, need_gene_count)?;

    let (lower_percentile, upper_percentile) =
        calculate_percentiles(&sum_genes_per_cell, &lower_lim, &upper_lim)?;

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

fn calculate_percentiles(
    values: &[f64],
    lower_lim: &FlexValue,
    upper_lim: &FlexValue,
) -> anyhow::Result<(f64, f64)> {
    let mut arr = Array1::from_vec(values.iter().map(|&x| n64(x)).collect());

    let lower_percentile = match lower_lim {
        FlexValue::Relative(p) => arr
            .quantile_axis_mut(Axis(0), n64(*p), &ndarray_stats::interpolate::Linear)
            .map_err(|e| anyhow::anyhow!("Error calculating lower percentile: {}", e))?
            .into_scalar()
            .raw(),
        _ => f64::MIN,
    };

    let upper_percentile = match upper_lim {
        FlexValue::Relative(p) => arr
            .quantile_axis_mut(Axis(0), n64(*p), &ndarray_stats::interpolate::Linear)
            .map_err(|e| anyhow::anyhow!("Error calculating upper percentile: {}", e))?
            .into_scalar()
            .raw(),
        _ => f64::MAX,
    };

    Ok((lower_percentile, upper_percentile))
}

fn calculate_gene_stats(
    adata: &IMAnnData,
    need_cell_count: bool,
) -> anyhow::Result<(Option<Vec<u32>>, Vec<f64>)> {
    let n_cells_per_gene = if need_cell_count {
        Some(crate::memory::statistics::compute_number(
            adata,
            Direction::Column,
        )?)
    } else {
        None
    };
    let sum_cells_per_gene = crate::memory::statistics::compute_sum(adata, Direction::Column)?;
    Ok((n_cells_per_gene, sum_cells_per_gene))
}

// TODO: refactor this and optimize code
fn create_gene_filter_mask(
    n_vars: usize,
    n_cells_per_gene: &Option<Vec<u32>>,
    sum_cells_per_gene: &[f64],
    lower_lim: &FlexValue,
    upper_lim: &FlexValue,
    lower_percentile: f64,
    upper_percentile: f64,
) -> Array1<bool> {
    Array1::from_vec(
        (0..n_vars)
            .map(|i| match (lower_lim, upper_lim) {
                (FlexValue::Absolute(lower), FlexValue::Absolute(upper)) => {
                    let n_cells = n_cells_per_gene.as_ref().unwrap()[i];
                    n_cells >= *lower && n_cells <= *upper
                }
                (FlexValue::Relative(_), FlexValue::Relative(_)) => {
                    let sum_cells = sum_cells_per_gene[i];
                    sum_cells >= lower_percentile && sum_cells <= upper_percentile
                }
                (FlexValue::Absolute(lower), FlexValue::Relative(_)) => {
                    let n_cells = n_cells_per_gene.as_ref().unwrap()[i];
                    let sum_cells = sum_cells_per_gene[i];
                    n_cells >= *lower && sum_cells <= upper_percentile
                }
                (FlexValue::Relative(_), FlexValue::Absolute(upper)) => {
                    let n_cells = n_cells_per_gene.as_ref().unwrap()[i];
                    let sum_cells = sum_cells_per_gene[i];
                    sum_cells >= lower_percentile && n_cells <= *upper
                }
                (FlexValue::Absolute(lower), FlexValue::None) => {
                    let n_cells = n_cells_per_gene.as_ref().unwrap()[i];
                    n_cells >= *lower
                }
                (FlexValue::None, FlexValue::Absolute(upper)) => {
                    let n_cells = n_cells_per_gene.as_ref().unwrap()[i];
                    n_cells <= *upper
                }
                (FlexValue::Relative(_), FlexValue::None) => {
                    let sum_cells = sum_cells_per_gene[i];
                    sum_cells >= lower_percentile
                }
                (FlexValue::None, FlexValue::Relative(_)) => {
                    let sum_cells = sum_cells_per_gene[i];
                    sum_cells <= upper_percentile
                }
                (FlexValue::None, FlexValue::None) => true,
            })
            .collect(),
    )
}

pub fn filter_genes_inplace(
    adata: &mut IMAnnData,
    lower_lim: FlexValue,
    upper_lim: FlexValue,
) -> anyhow::Result<()> {
    let need_cell_count =
        matches!(lower_lim, FlexValue::Absolute(_)) || matches!(upper_lim, FlexValue::Absolute(_));
    let (n_cells_per_gene, sum_cells_per_gene) = calculate_gene_stats(adata, need_cell_count)?;

    let (lower_percentile, upper_percentile) =
        calculate_percentiles(&sum_cells_per_gene, &lower_lim, &upper_lim)?;

    let mask = create_gene_filter_mask(
        adata.n_vars(),
        &n_cells_per_gene,
        &sum_cells_per_gene,
        &lower_lim,
        &upper_lim,
        lower_percentile,
        upper_percentile,
    );

    let selection = crate::shared::processing::get_select_info_vars(Some(mask.view()))?;
    let selection_refs: Vec<&SelectInfoElem> = selection.iter().collect();

    adata.subset_inplace(selection_refs.as_slice())
}

pub fn filter_genes(
    adata: &IMAnnData,
    lower_lim: FlexValue,
    upper_lim: FlexValue,
) -> anyhow::Result<IMAnnData> {
    let need_cell_count =
        matches!(lower_lim, FlexValue::Absolute(_)) || matches!(upper_lim, FlexValue::Absolute(_));
    let (n_cells_per_gene, sum_cells_per_gene) = calculate_gene_stats(adata, need_cell_count)?;

    let (lower_percentile, upper_percentile) =
        calculate_percentiles(&sum_cells_per_gene, &lower_lim, &upper_lim)?;

    let mask = create_gene_filter_mask(
        adata.n_vars(),
        &n_cells_per_gene,
        &sum_cells_per_gene,
        &lower_lim,
        &upper_lim,
        lower_percentile,
        upper_percentile,
    );

    let selection = crate::shared::processing::get_select_info_vars(Some(mask.view()))?;
    let selection_refs: Vec<&SelectInfoElem> = selection.iter().collect();

    adata.subset(selection_refs.as_slice())
}

/// normalization of the data

pub fn normalize_total_inplace(
    adata: &mut IMAnnData,
    target_sum: f64,
    direction: Direction,
) -> anyhow::Result<()> {
    match direction {
        Direction::Row => scale::scale_row(adata, target_sum),
        Direction::Column => scale::scale_col(adata, target_sum),
    }
}

pub fn normalize_total(
    adata: &IMAnnData,
    target_sum: f64,
    direction: Direction,
) -> anyhow::Result<IMAnnData> {
    let mut new_data = adata.deep_clone();
    normalize_total_inplace(&mut new_data, target_sum, direction)?;
    Ok(new_data)
}

pub fn log1p_transform_inplace(adata: &mut IMAnnData) -> anyhow::Result<()> {
    transform::log1p_data(adata)
}

pub fn log1p_transform(adata: &IMAnnData) -> anyhow::Result<IMAnnData> {
    let mut new_data = adata.deep_clone();
    log1p_transform_inplace(&mut new_data)?;
    Ok(new_data)
}

#[cfg(test)]
mod tests {

    use super::*;
    use anndata::{data::DynCsrMatrix, ArrayData};
    use anndata_memory::IMAnnData;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};
    use rand::{distributions::Uniform, prelude::Distribution, Rng};

    fn create_large_test_data(
        nrows: usize,
        ncols: usize,
        sparsity: f64,
    ) -> (ArrayData, Vec<String>, Vec<String>) {
        let mut rng = rand::thread_rng();
        let value_dist = Uniform::new(0.0, 50.0);

        // Create a COO matrix
        let mut coo_matrix = CooMatrix::new(nrows, ncols);

        // Determine the number of non-zero elements
        let nnz = (nrows * ncols) as f64 * (1.0 / sparsity);
        println!("Number of non-zero elements: {}", nnz);
        let nnz = nnz.round() as usize;

        // Add non-zero elements
        for _ in 0..nnz {
            let row = rng.gen_range(0..nrows);
            let col = rng.gen_range(0..ncols);
            let value = value_dist.sample(&mut rng);
            coo_matrix.push(row, col, value);
        }

        // Convert to CSR format
        let csr_matrix: CsrMatrix<f64> = CsrMatrix::from(&coo_matrix);
        let matrix = DynCsrMatrix::from(csr_matrix);

        // Generate obs and var names
        let obs_names = (0..nrows).map(|i| format!("obs{}", i)).collect();
        let var_names = (0..ncols).map(|i| format!("var{}", i)).collect();

        (ArrayData::CsrMatrix(matrix), obs_names, var_names)
    }

    // Function to create a large test AnnData object
    fn create_large_test_anndata(nrows: usize, ncols: usize, sparsity: f64) -> IMAnnData {
        let (x, obs_names, var_names) = create_large_test_data(nrows, ncols, sparsity);
        IMAnnData::new_basic(x, obs_names, var_names).unwrap()
    }

    #[test]
    fn test_filter_cells() -> anyhow::Result<()> {
        let adata = create_large_test_anndata(1000, 100, 10.0);

        // Test filtering cells with absolute values
        let filtered = filter_cells(&adata, FlexValue::Absolute(5), FlexValue::Absolute(15))?;
        println!("Cells after absolute filtering: {}", filtered.n_obs());
        assert!(filtered.n_obs() < adata.n_obs());

        // Test filtering cells with relative values
        let filtered = filter_cells(&adata, FlexValue::Relative(0.1), FlexValue::Relative(0.9))?;
        println!("Cells after relative filtering: {}", filtered.n_obs());
        assert!(filtered.n_obs() < adata.n_obs());

        Ok(())
    }

    #[test]
    fn test_filter_genes() -> anyhow::Result<()> {
        let adata = create_large_test_anndata(1000, 100, 10.0);

        // Test filtering genes with absolute values
        let filtered = filter_genes(&adata, FlexValue::Absolute(50), FlexValue::Absolute(100))?;
        println!("Genes before filtering: {}", adata.n_vars());
        println!("Genes after absolute filtering: {}", filtered.n_vars());
        assert!(filtered.n_vars() < adata.n_vars());

        // Test filtering genes with relative values
        let filtered = filter_genes(&adata, FlexValue::Relative(0.1), FlexValue::Relative(0.9))?;
        println!("Genes after relative filtering: {}", filtered.n_vars());
        assert!(filtered.n_vars() < adata.n_vars());

        Ok(())
    }

    #[test]
    fn test_normalize_total() -> anyhow::Result<()> {
        let adata = create_large_test_anndata(1000, 100, 10.0);
        let target_sum = 1e4;

        // Test row-wise normalization
        let normalized = normalize_total(&adata, target_sum, Direction::Row)?;
        let x = normalized.x().get_data()?;
        if let ArrayData::CsrMatrix(dyn_csr) = x {
            match dyn_csr {
                DynCsrMatrix::F64(csr) => check_row_sums(&csr, target_sum)?,
                _ => panic!("Unexpected CsrMatrix type"),
            }
        } else {
            panic!("Expected CSR matrix");
        }

        // Test column-wise normalization
        let normalized = normalize_total(&adata, target_sum, Direction::Column)?;
        let x = normalized.x().get_data()?;
        if let ArrayData::CsrMatrix(dyn_csr) = x {
            match dyn_csr {
                DynCsrMatrix::F64(csr) => check_column_sums(&csr, target_sum)?,
                _ => panic!("Unexpected CsrMatrix type"),
            }
        } else {
            panic!("Expected CSR matrix");
        }

        Ok(())
    }

    fn check_row_sums(csr: &CsrMatrix<f64>, target_sum: f64) -> anyhow::Result<()> {
        for row in csr.row_iter() {
            let row_sum: f64 = row.values().iter().sum();
            assert!(
                (row_sum - target_sum).abs() < 1e-6,
                "Row sum {} does not match target sum {}",
                row_sum, 
                target_sum
            );
        }
        Ok(())
    }

    fn check_column_sums(csr: &CsrMatrix<f64>, target_sum: f64) -> anyhow::Result<()> {
        let mut column_sums = vec![0.0; csr.ncols()];
        for row in csr.row_iter() {
            for (&col_idx, &value) in row.col_indices().iter().zip(row.values()) {
                column_sums[col_idx] += value;
            }
        }
        for (col_idx, &sum) in column_sums.iter().enumerate() {
            assert!(
                (sum - target_sum).abs() < 1e-6,
                "Column {} sum {} does not match target sum {}",
                col_idx,
                sum,
                target_sum
            );
        }
        Ok(())
    }
}
