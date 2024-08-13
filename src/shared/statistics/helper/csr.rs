use std::iter::Sum;

use anndata::data::DynCsrMatrix;
use nalgebra_sparse::CsrMatrix;

use crate::{
    match_dyn_csr_matrix,
    shared::{Direction, NumericOps},
};

pub(crate) fn number_whole(csr: &DynCsrMatrix, direction: Direction) -> anyhow::Result<Vec<u32>> {
    match_dyn_csr_matrix!(csr, number_whole_helper, direction)
}

/// Computes the number of non-zero entries in the defined direction (row/column wise)
fn number_whole_helper<T: NumericOps>(
    csr: &CsrMatrix<T>,
    direction: Direction,
) -> anyhow::Result<Vec<u32>> {
    match direction {
        Direction::Row => {
            // For row-wise computation
            Ok(csr
                .row_offsets()
                .windows(2)
                .map(|window| (window[1] - window[0]) as u32)
                .collect())
        }
        Direction::Column => {
            // For column-wise computation
            let mut result = vec![0; csr.ncols()];
            for &col_index in csr.col_indices() {
                result[col_index] += 1;
            }
            Ok(result)
        }
    }
}

pub(crate) fn number_chunk(
    csr: &DynCsrMatrix,
    direction: &Direction,
    reference: &mut Vec<u32>,
) -> anyhow::Result<()> {
    match_dyn_csr_matrix!(csr, number_chunk_helper, direction, reference)
}

fn number_chunk_helper<T: NumericOps>(
    csr: &CsrMatrix<T>,
    direction: &Direction,
    reference: &mut Vec<u32>,
) -> anyhow::Result<()> {
    match direction {
        Direction::Row => {
            // For row-wise computation
            for (i, window) in csr.row_offsets().windows(2).enumerate() {
                let count = (window[1] - window[0]) as u32;
                if i < reference.len() {
                    reference[i] += count;
                }
            }
        }
        Direction::Column => {
            // For column-wise computation
            for &col_index in csr.col_indices() {
                if col_index < reference.len() {
                    reference[col_index] += 1;
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn sum_whole(csr: &DynCsrMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    match_dyn_csr_matrix!(csr, sum_whole_helper, direction)
}

/// Computes the sum of entries in the defined direction (row/column wise)
fn sum_whole_helper<T>(csr: &CsrMatrix<T>, direction: Direction) -> anyhow::Result<Vec<f64>>
where
    T: NumericOps,
    f64: From<T>,
{
    match direction {
        Direction::Row => {
            let mut result = vec![0.0; csr.nrows()];
            for (row, row_vec) in csr.row_iter().enumerate() {
                result[row] = row_vec.values().iter().map(|&v| f64::from(v)).sum();
            }
            Ok(result)
        }
        Direction::Column => {
            let mut result = vec![0.0; csr.ncols()];
            for (&col_index, &value) in csr.col_indices().iter().zip(csr.values().iter()) {
                result[col_index] += f64::from(value);
            }
            Ok(result)
        }
    }
}

pub(crate) fn sum_chunk(
    csr: &DynCsrMatrix,
    direction: &Direction,
    reference: &mut Vec<f64>,
) -> anyhow::Result<()> {
    match_dyn_csr_matrix!(csr, sum_chunk_helper, direction, reference)
}

fn sum_chunk_helper<T>(
    csr: &CsrMatrix<T>,
    direction: &Direction,
    reference: &mut Vec<f64>,
) -> anyhow::Result<()>
where
    T: NumericOps + Sum,
    f64: From<T>,
{
    
    match direction {
        Direction::Row => {
            // For row-wise computation
            for (row, row_vec) in csr.row_iter().enumerate() {
                reference[row] = row_vec.values().iter().map(|&v| f64::from(v)).sum();
            }
        }
        Direction::Column => {
            // For column-wise computation
            for (&col_index, &value) in csr
                .col_indices()
                .iter()
                .zip(csr.values().iter())
            {
                if col_index < reference.len() {
                    reference[col_index] += f64::from(value);
                }
            }
        }
    }
    Ok(())
}
