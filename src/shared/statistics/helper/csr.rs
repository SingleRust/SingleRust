use std::{iter::Sum, sync::atomic::{AtomicU32, Ordering}};

use anndata::data::DynCsrMatrix;
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSlice};

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
    T: NumericOps + Zero,
    f64: From<T>,
{
    let (row_offsets, col_indices, values) = csr.csr_data();

    match direction {
        Direction::Row => {
            let mut result = vec![T::zero(); csr.nrows()];
            for i in 0..csr.nrows() {
                let start = row_offsets[i];
                let end = row_offsets[i + 1];
                result[i] = values[start..end].iter().fold(T::zero(), |acc, &x| acc + x);
            }
            Ok(result.into_iter().map(f64::from).collect())
        }
        Direction::Column => {
            let mut result = vec![T::zero(); csr.ncols()];
            for (&col_index, &value) in col_indices.iter().zip(values.iter()) {
                result[col_index] += value;
            }
            Ok(result.into_iter().map(f64::from).collect())
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

pub(crate) fn variance_whole(csr: &DynCsrMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    match_dyn_csr_matrix!(csr, variance_whole_helper, direction)
}

fn variance_whole_helper<T>(csr: &CsrMatrix<T>, direction: Direction) -> anyhow::Result<Vec<f64>>
where
    T: NumericOps,
    f64: From<T>,
{
    let sum = sum_whole_helper(csr, direction.clone())?;
    let count = number_whole_helper(csr, direction.clone())?;
    
    match direction {
        Direction::Row => {
            let mut result = vec![0.0; csr.nrows()];
            for (row, row_vec) in csr.row_iter().enumerate() {
                let mean = sum[row] / count[row] as f64;
                let variance = row_vec.values().iter()
                    .map(|&v| {
                        let diff = f64::from(v) - mean;
                        diff * diff
                    })
                    .sum::<f64>() / count[row] as f64;
                result[row] = variance;
            }
            Ok(result)
        }
        Direction::Column => {
            let mut result = vec![0.0; csr.ncols()];
            let mut squared_sum = vec![0.0; csr.ncols()];
            for (&col_index, &value) in csr.col_indices().iter().zip(csr.values().iter()) {
                let val = f64::from(value);
                squared_sum[col_index] += val * val;
            }
            for col in 0..csr.ncols() {
                if count[col] > 0 {
                    let mean = sum[col] / count[col] as f64;
                    result[col] = squared_sum[col] / count[col] as f64 - mean * mean;
                }
            }
            Ok(result)
        }
    }
}

pub(crate) fn min_max_whole(csr: &DynCsrMatrix, direction: Direction) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match_dyn_csr_matrix!(csr, min_max_whole_helper, direction)
}

fn min_max_whole_helper<T>(csr: &CsrMatrix<T>, direction: Direction) -> anyhow::Result<(Vec<f64>, Vec<f64>)>
where
    T: NumericOps,
    f64: From<T>,
{
    match direction {
        Direction::Row => {
            let mut min_values = vec![f64::INFINITY; csr.nrows()];
            let mut max_values = vec![f64::NEG_INFINITY; csr.nrows()];
            for (row, row_vec) in csr.row_iter().enumerate() {
                for &value in row_vec.values() {
                    let val = f64::from(value);
                    min_values[row] = min_values[row].min(val);
                    max_values[row] = max_values[row].max(val);
                }
            }
            Ok((min_values, max_values))
        }
        Direction::Column => {
            let mut min_values = vec![f64::INFINITY; csr.ncols()];
            let mut max_values = vec![f64::NEG_INFINITY; csr.ncols()];
            for (&col_index, &value) in csr.col_indices().iter().zip(csr.values().iter()) {
                let val = f64::from(value);
                min_values[col_index] = min_values[col_index].min(val);
                max_values[col_index] = max_values[col_index].max(val);
            }
            Ok((min_values, max_values))
        }
    }
}

pub(crate) fn std_dev_whole(csr: &DynCsrMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let variances = variance_whole(csr, direction)?;
    Ok(variances.into_iter().map(|v| v.sqrt()).collect())
}