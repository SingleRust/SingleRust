use anndata::data::DynCscMatrix;
use nalgebra_sparse::CscMatrix;
use num_traits::Zero;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::{
    match_dyn_csc_matrix,
    shared::{Direction, NumericOps},
};

pub(crate) fn number_whole(csc: &DynCscMatrix, direction: Direction) -> anyhow::Result<Vec<u32>> {
    match_dyn_csc_matrix!(csc, number_whole_helper, direction)
}

fn number_whole_helper<T: NumericOps>(
    csc: &CscMatrix<T>,
    direction: Direction,
) -> anyhow::Result<Vec<u32>> {
    match direction {
        Direction::Row => {
            let mut result = vec![0; csc.nrows()];
            for &row_index in csc.row_indices() {
                result[row_index] += 1;
            }
            Ok(result)
        }
        Direction::Column => {
            Ok(csc
                .col_offsets()
                .par_windows(2)
                .map(|window| (window[1] - window[0]) as u32)
                .collect())
        }
    }
}

pub(crate) fn number_chunk(
    csc: &DynCscMatrix,
    direction: &Direction,
    reference: &mut Vec<u32>,
) -> anyhow::Result<()> {
    match_dyn_csc_matrix!(csc, number_chunk_helper, direction, reference)
}

fn number_chunk_helper<T: NumericOps>(
    csc: &CscMatrix<T>,
    direction: &Direction,
    reference: &mut Vec<u32>,
) -> anyhow::Result<()> {
    match direction {
        Direction::Column => {
            for (i, window) in csc.col_offsets().windows(2).enumerate() {
                let count = (window[1] - window[0]) as u32;
                if i < reference.len() {
                    reference[i] += count;
                }
            }
        }
        Direction::Row => {
            for &row_index in csc.row_indices() {
                if row_index < reference.len() {
                    reference[row_index] += 1;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn sum_whole(csc: &DynCscMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    match_dyn_csc_matrix!(csc, sum_whole_helper, direction)
}

fn sum_whole_helper<T>(csc: &CscMatrix<T>, direction: Direction) -> anyhow::Result<Vec<f64>>
where
    T: NumericOps + Zero,
    f64: From<T>,
{
    let (col_offsets, row_indices, values) = csc.csc_data();

    match direction {
        Direction::Row => {
            let mut result = vec![T::zero(); csc.nrows()];
            for (&row_index, &value) in row_indices.iter().zip(values.iter()) {
                result[row_index] += value;
            }
            Ok(result.into_iter().map(f64::from).collect())
        }
        Direction::Column => {
            let mut result = vec![T::zero(); csc.ncols()];
            for i in 0..csc.ncols() {
                let start = col_offsets[i];
                let end = col_offsets[i + 1];
                result[i] = values[start..end].iter().fold(T::zero(), |acc, &x| acc + x);
            }
            Ok(result.into_iter().map(f64::from).collect())
        }
    }
}

pub(crate) fn sum_chunk(
    csc: &DynCscMatrix,
    direction: &Direction,
    reference: &mut Vec<f64>,
) -> anyhow::Result<()> {
    match_dyn_csc_matrix!(csc, sum_chunk_helper, direction, reference)
}

fn sum_chunk_helper<T>(
    csc: &CscMatrix<T>,
    direction: &Direction,
    reference: &mut Vec<f64>,
) -> anyhow::Result<()>
where
    T: NumericOps,
    f64: From<T>,
{
    match direction {
        Direction::Column => {
            for (col, col_vec) in csc.col_iter().enumerate() {
                if col < reference.len() {
                    reference[col] += col_vec.values().iter().map(|&v| f64::from(v)).sum::<f64>();
                }
            }
        }
        Direction::Row => {
            for (&row_index, &value) in csc.row_indices().iter().zip(csc.values().iter()) {
                if row_index < reference.len() {
                    reference[row_index] += f64::from(value);
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn variance_whole(csc: &DynCscMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    match_dyn_csc_matrix!(csc, variance_whole_helper, direction)
}

fn variance_whole_helper<T>(csc: &CscMatrix<T>, direction: Direction) -> anyhow::Result<Vec<f64>>
where
    T: NumericOps,
    f64: From<T>,
{
    let sum = sum_whole_helper(csc, direction.clone())?;
    let count = number_whole_helper(csc, direction.clone())?;
    
    match direction {
        Direction::Row => {
            let mut result = vec![0.0; csc.nrows()];
            let mut squared_sum = vec![0.0; csc.nrows()];
            for (&row_index, &value) in csc.row_indices().iter().zip(csc.values().iter()) {
                let val = f64::from(value);
                squared_sum[row_index] += val * val;
            }
            for row in 0..csc.nrows() {
                if count[row] > 0 {
                    let mean = sum[row] / count[row] as f64;
                    result[row] = squared_sum[row] / count[row] as f64 - mean * mean;
                }
            }
            Ok(result)
        }
        Direction::Column => {
            let mut result = vec![0.0; csc.ncols()];
            for (col, col_vec) in csc.col_iter().enumerate() {
                let mean = sum[col] / count[col] as f64;
                let variance = col_vec.values().iter()
                    .map(|&v| {
                        let diff = f64::from(v) - mean;
                        diff * diff
                    })
                    .sum::<f64>() / count[col] as f64;
                result[col] = variance;
            }
            Ok(result)
        }
    }
}

pub(crate) fn min_max_whole(csc: &DynCscMatrix, direction: Direction) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match_dyn_csc_matrix!(csc, min_max_whole_helper, direction)
}

fn min_max_whole_helper<T>(csc: &CscMatrix<T>, direction: Direction) -> anyhow::Result<(Vec<f64>, Vec<f64>)>
where
    T: NumericOps,
    f64: From<T>,
{
    match direction {
        Direction::Row => {
            let mut min_values = vec![f64::INFINITY; csc.nrows()];
            let mut max_values = vec![f64::NEG_INFINITY; csc.nrows()];
            for (&row_index, &value) in csc.row_indices().iter().zip(csc.values().iter()) {
                let val = f64::from(value);
                min_values[row_index] = min_values[row_index].min(val);
                max_values[row_index] = max_values[row_index].max(val);
            }
            Ok((min_values, max_values))
        }
        Direction::Column => {
            let mut min_values = vec![f64::INFINITY; csc.ncols()];
            let mut max_values = vec![f64::NEG_INFINITY; csc.ncols()];
            for (col, col_vec) in csc.col_iter().enumerate() {
                for &value in col_vec.values() {
                    let val = f64::from(value);
                    min_values[col] = min_values[col].min(val);
                    max_values[col] = max_values[col].max(val);
                }
            }
            Ok((min_values, max_values))
        }
    }
}

pub(crate) fn std_dev_whole(csc: &DynCscMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let variances = variance_whole(csc, direction)?;
    Ok(variances.into_iter().map(|v| v.sqrt()).collect())
}