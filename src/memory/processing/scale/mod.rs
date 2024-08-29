use std::ops::DerefMut;
use anndata::ArrayData;
use anndata_memory::IMAnnData;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use anndata::data::{DynCscMatrix, DynCsrMatrix};

pub(crate) fn scale_row(adata: &mut IMAnnData, target_value: f64) -> anyhow::Result<()> {
    let sum_row = crate::memory::statistics::compute_sum(adata, crate::Direction::Row)?;
    let scale: Vec<f64> = sum_row.iter().map(|sum| {
        if *sum == 0.0 {
            0.0
        } else {
            target_value / *sum
        }
    }).collect();

    let x = adata.x();
    match x.get_type()? {
        anndata::backend::DataType::CsrMatrix(_) => scale_row_csr(adata, &scale),
        anndata::backend::DataType::CscMatrix(_) => scale_row_csc(adata, &scale),
        _ => Err(anyhow::anyhow!("Unsupported data type for scaling")),
    }
}

fn scale_row_csc(adata: &mut IMAnnData, scale: &[f64]) -> anyhow::Result<()> {
    let x_data = adata.x();
    let mut write_guard = x_data.0.write_inner();
    let arr_data = write_guard.deref_mut();

    if let ArrayData::CscMatrix(ref mut csc_matrix) = arr_data {
        match csc_matrix {
            DynCscMatrix::F64(matrix) => {
                let (col_offsets, row_indices, values) = matrix.csc_data_mut();
                for col in 0..col_offsets.len() - 1 {
                    for j in col_offsets[col]..col_offsets[col + 1] {
                        let row = row_indices[j];
                        values[j] *= scale[row];
                    }
                }
            },
            _ => {
                let mut float_matrix: CscMatrix<f64> = csc_matrix.clone().try_into()?;
                let (col_offsets, row_indices, values) = float_matrix.csc_data_mut();
                for col in 0..col_offsets.len() - 1 {
                    for j in col_offsets[col]..col_offsets[col + 1] {
                        let row = row_indices[j];
                        values[j] *= scale[row];
                    }
                }
                *csc_matrix = DynCscMatrix::F64(float_matrix);
            }
        }
        Ok(())
    } else {
        Err(anyhow::anyhow!("X is not a CSC matrix"))
    }
}

fn scale_row_csr(adata: &mut IMAnnData, scale: &[f64]) -> anyhow::Result<()> {
    let x_data = adata.x();
    let mut write_guard = x_data.0.write_inner();
    let arr_data = write_guard.deref_mut();

    if let ArrayData::CsrMatrix(ref mut csr_matrix) = arr_data {
        match csr_matrix {
            DynCsrMatrix::F64(matrix) => {
                for (row_index, mut row) in matrix.row_iter_mut().enumerate() {
                    let row_scale = scale[row_index];
                    for val in row.values_mut() {
                        *val *= row_scale;
                    }
                }
            },
            _ => {
                let mut float_matrix: CsrMatrix<f64> = csr_matrix.clone().try_into()?;
                for (row_index, mut row) in float_matrix.row_iter_mut().enumerate() {
                    let row_scale = scale[row_index];
                    for val in row.values_mut() {
                        *val *= row_scale;
                    }
                }
                *csr_matrix = DynCsrMatrix::F64(float_matrix);
            }
        }
        Ok(())
    } else {
        Err(anyhow::anyhow!("X is not a CSR matrix"))
    }
}

pub(crate) fn scale_col(adata: &mut IMAnnData, target_value: f64) -> anyhow::Result<()> {
    let sum_col = crate::memory::statistics::compute_sum(adata, crate::Direction::Column)?;
    let scale: Vec<f64> = sum_col.iter().map(|sum| {
        if *sum == 0.0 {
            0.0
        } else {
            target_value / *sum
        }
    }).collect();

    let x = adata.x();
    match x.get_type()? {
        anndata::backend::DataType::CsrMatrix(_) => scale_col_csr(adata, &scale),
        anndata::backend::DataType::CscMatrix(_) => scale_col_csc(adata, &scale),
        _ => Err(anyhow::anyhow!("Unsupported data type for scaling")),
    }
}

fn scale_col_csc(adata: &mut IMAnnData, scale: &[f64]) -> anyhow::Result<()> {
    let x_data = adata.x();
    let mut write_guard = x_data.0.write_inner();
    let arr_data = write_guard.deref_mut();

    if let ArrayData::CscMatrix(ref mut csc_matrix) = arr_data {
        match csc_matrix {
            DynCscMatrix::F64(matrix) => {
                let (col_offsets, _row_indices, values) = matrix.csc_data_mut();
                for col in 0..col_offsets.len() - 1 {
                    for j in col_offsets[col]..col_offsets[col + 1] {
                        values[j] *= scale[col];
                    }
                }
            },
            _ => {
                let mut float_matrix: CscMatrix<f64> = csc_matrix.clone().try_into()?;
                let (col_offsets, _row_indices, values) = float_matrix.csc_data_mut();
                for col in 0..col_offsets.len() - 1 {
                    for j in col_offsets[col]..col_offsets[col + 1] {
                        values[j] *= scale[col];
                    }
                }
                *csc_matrix = DynCscMatrix::F64(float_matrix);
            }
        }
        Ok(())
    } else {
        Err(anyhow::anyhow!("X is not a CSC matrix"))
    }
}

fn scale_col_csr(adata: &mut IMAnnData, scale: &[f64]) -> anyhow::Result<()> {
    let x_data = adata.x();
    let mut write_guard = x_data.0.write_inner();
    let arr_data = write_guard.deref_mut();

    if let ArrayData::CsrMatrix(ref mut csr_matrix) = arr_data {
        match csr_matrix {
            DynCsrMatrix::F64(matrix) => {
                let (row_offsets, col_indices, values) = matrix.csr_data_mut();
                for row in 0..row_offsets.len() - 1 {
                    for j in row_offsets[row]..row_offsets[row + 1] {
                        let col = col_indices[j];
                        values[j] *= scale[col];
                    }
                }
            },
            _ => {
                let mut float_matrix: CsrMatrix<f64> = csr_matrix.clone().try_into()?;
                let (row_offsets, col_indices, values) = float_matrix.csr_data_mut();
                for row in 0..row_offsets.len() - 1 {
                    for j in row_offsets[row]..row_offsets[row + 1] {
                        let col = col_indices[j];
                        values[j] *= scale[col];
                    }
                }
                *csr_matrix = DynCsrMatrix::F64(float_matrix);
            }
        }
        Ok(())
    } else {
        Err(anyhow::anyhow!("X is not a CSR matrix"))
    }
}