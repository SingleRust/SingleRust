use std::iter::Sum;

use anndata::data::DynCscMatrix;
use nalgebra_sparse::CscMatrix;

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
                .windows(2)
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
    T: NumericOps,
    f64: From<T>,
{
    match direction {
        Direction::Row => {
            let mut result = vec![0.0; csc.nrows()];
            for (&row_index, &value) in csc.row_indices().iter().zip(csc.values().iter()) {
                result[row_index] += f64::from(value);
            }
            Ok(result)
        }
        Direction::Column => {
            let mut result = vec![0.0; csc.ncols()];
            for (col, col_vec) in csc.col_iter().enumerate() {
                result[col] = col_vec.values().iter().map(|&v| f64::from(v)).sum();
            }
            Ok(result)
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
    T: NumericOps + Sum,
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