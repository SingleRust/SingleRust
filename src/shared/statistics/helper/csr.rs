use anndata::data::DynCsrMatrix;
use nalgebra_sparse::CsrMatrix;

use crate::shared::Direction;

/// Computes the number of non-zero entries in the defined direction (row/column wise)
pub(crate) fn number_whole(csr: DynCsrMatrix, direction: Direction) -> anyhow::Result<Vec<u32>> {
    let converted_csr: CsrMatrix<f64> = csr.try_into()?;

    match direction {
        Direction::Row => {
            // For row-wise computation
            Ok(converted_csr
                .row_offsets()
                .windows(2)
                .map(|window| (window[1] - window[0]) as u32)
                .collect())
        }
        Direction::Column => {
            // For column-wise computation
            let mut result = vec![0; converted_csr.ncols()];
            for &col_index in converted_csr.col_indices() {
                result[col_index] += 1;
            }
            Ok(result)
        }
    }
}

pub(crate) fn number_chunk(
    csr: DynCsrMatrix,
    direction: &Direction,
    reference: &mut Vec<u32>,
) -> anyhow::Result<()> {
    let converted_csr: CsrMatrix<f64> = csr.try_into()?;

    match direction {
        Direction::Row => {
            // For row-wise computation
            for (i, window) in converted_csr.row_offsets().windows(2).enumerate() {
                let count = (window[1] - window[0]) as u32;
                if i < reference.len() {
                    reference[i] += count;
                }
            }
        }
        Direction::Column => {
            // For column-wise computation
            for &col_index in converted_csr.col_indices() {
                if col_index < reference.len() {
                    reference[col_index] += 1;
                }
            }
        }
    }

    Ok(())
}

/// Computes the sum of entries in the defined direction (row/column wise)
pub(crate) fn sum_whole(csr: DynCsrMatrix, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let converted_csr: CsrMatrix<f64> = csr.try_into()?;
    match direction {
        Direction::Row => {
            let mut result = vec![0.0; converted_csr.nrows()];
            for (row, row_vec) in converted_csr.row_iter().enumerate() {
                result[row] = row_vec.values().iter().sum();
            }
            Ok(result)
        }
        Direction::Column => {
            let mut result = vec![0.0; converted_csr.ncols()];
            for (&col_index, &value) in converted_csr
                .col_indices()
                .iter()
                .zip(converted_csr.values().iter())
            {
                result[col_index] += value;
            }
            Ok(result)
        }
    }
}

// TODO !!! refactor this !!!!!
pub(crate) fn sum_chunk(
    csr: DynCsrMatrix,
    direction: &Direction,
    reference: &mut Vec<f64>,
) -> anyhow::Result<()> {
    let converted_csr: CsrMatrix<f64> = csr.try_into()?;
    match direction {
        Direction::Row => {
            // For row-wise computation
            for (row, row_vec) in converted_csr.row_iter().enumerate() {
                reference[row] = row_vec.values().iter().sum();
            }
        }
        Direction::Column => {
            // For column-wise computation
            for (&col_index, &value) in converted_csr
                .col_indices()
                .iter()
                .zip(converted_csr.values().iter())
            {
                if col_index < reference.len() {
                    reference[col_index] += value;
                }
            }
        }
    }
    Ok(())
}
