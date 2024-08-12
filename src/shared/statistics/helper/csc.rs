use anndata::data::DynCscMatrix;
use nalgebra_sparse::CscMatrix;

use crate::shared::Direction;

/// computes the number of non.zero entries in the definted direction (row/column wise)
pub(crate) fn number_whole(csc: DynCscMatrix, direction: Direction) -> anyhow::Result<Vec<u32>> {
    let converted_csc: CscMatrix<f64> = csc.try_into()?;

    match direction {
        Direction::Row => {
            let mut result = vec![0; converted_csc.nrows()];
            for &row_index in converted_csc.row_indices() {
                result[row_index] += 1;
            }
            Ok(result)
        }
        Direction::Column => Ok(converted_csc
            .col_offsets()
            .windows(2)
            .map(|window| (window[1] - window[0]) as u32)
            .collect()),
    }
}

pub(crate) fn number_chunk(
    csc: DynCscMatrix,
    direction: &Direction,
    reference: &mut Vec<u32>,
) -> anyhow::Result<()> {
    let converted_csc: CscMatrix<f64> = csc.try_into()?;

    match direction {
        Direction::Column => {
            // For column-wise computation
            for (i, window) in converted_csc.col_offsets().windows(2).enumerate() {
                let count = (window[1] - window[0]) as u32;
                if i < reference.len() {
                    reference[i] += count;
                }
            }
        }
        Direction::Row => {
            // For row-wise computation
            for &row_index in converted_csc.row_indices() {
                if row_index < reference.len() {
                    reference[row_index] += 1;
                }
            }
        }
    }

    Ok(())
}

/// Computes the sum of entries in the defined direction (row/column wise)
pub(crate) fn sum_whole(csc: DynCscMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let converted_csc: CscMatrix<f64> = csc.try_into()?;
    match direction {
        Direction::Row => {
            let mut result = vec![0.0; converted_csc.nrows()];
            for (&row_index, &value) in converted_csc
                .row_indices()
                .iter()
                .zip(converted_csc.values().iter())
            {
                result[row_index] += value;
            }
            Ok(result)
        }
        Direction::Column => {
            let mut result = vec![0.0; converted_csc.ncols()];
            for (col, col_vec) in converted_csc.col_iter().enumerate() {
                result[col] = col_vec.values().iter().sum();
            }
            Ok(result)
        }
    }
}

// TODO !!! refactor this !!!!!

pub(crate) fn sum_chunk(
    csc: DynCscMatrix,
    direction: &Direction,
    reference: &mut Vec<f64>,
) -> anyhow::Result<()> {
    let converted_csc: CscMatrix<f64> = csc.try_into()?;
    match direction {
        Direction::Column => {
            // For column-wise computation
            for (col, col_vec) in converted_csc.col_iter().enumerate() {
                reference[col] = col_vec.values().iter().sum();
            }
        }
        Direction::Row => {
            // For row-wise computation
            for (&row_index, &value) in converted_csc
                .row_indices()
                .iter()
                .zip(converted_csc.values().iter())
            {
                if row_index < reference.len() {
                    reference[row_index] += value;
                }
            }
        }
    }
    Ok(())
}
