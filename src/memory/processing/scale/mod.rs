use std::ops::DerefMut;

use anndata::ArrayData;
use anndata_memory::IMAnnData;
use nalgebra_sparse::{CscMatrix, CsrMatrix};


pub(crate) fn scale_row(adata: &mut IMAnnData, target_value: f64) -> anyhow::Result<()> {

    let sum_row = crate::memory::statistics::compute_sum(adata, crate::Direction::Row)?;
    let non_zero_row = crate::memory::statistics::compute_number(adata, crate::Direction::Row)?;
    let scale = sum_row.iter().zip(non_zero_row.iter()).map(|(sum, non_zero)| {
        if *non_zero == 0 {
            0.0
        } else {
            target_value / (*sum / *non_zero as f64)
        }
    }).collect::<Vec<f64>>();

    let x = adata.x();

    match x.get_type()? {
        anndata::backend::DataType::Array(_) => todo!(),
        anndata::backend::DataType::Categorical => todo!(),
        anndata::backend::DataType::CsrMatrix(_) => scale_row_csc(adata, scale),
        anndata::backend::DataType::CscMatrix(_) => scale_row_csr(adata, scale),
        anndata::backend::DataType::DataFrame => todo!(),
        anndata::backend::DataType::Scalar(_) => todo!(),
        anndata::backend::DataType::Mapping => todo!(),
    }

}

fn scale_row_csc(adata: &mut IMAnnData, scale: Vec<f64>) -> anyhow::Result<()> {
    let x_data = adata.x();
    let read_guard = x_data.0.read_inner();
    let arr_data_clone = read_guard.clone();
    let mut float_data: CscMatrix<f64> = arr_data_clone.try_into().expect("Failed to convert to CscMatrix<f64>");

    let (col_offsets, row_indices, values) = float_data.csc_data_mut();
    for col in 0..col_offsets.len() - 1 {
        for j in col_offsets[col]..col_offsets[col + 1] {
            let row = row_indices[j];
            values[j] *= scale[row];
        }
    }
    drop(read_guard);
    let _ = x_data.0.insert(ArrayData::from(float_data));
    Ok(())
}

fn scale_row_csr(adata: &mut IMAnnData, scale: Vec<f64>) -> anyhow::Result<()> {
    let x_data = adata.x();
    let read_guard = x_data.0.read_inner();
    let arr_data_clone = read_guard.clone();
    let mut float_data: CsrMatrix<f64> = arr_data_clone.try_into().expect("Failed to convert to CsrMatrix<f64>");

    for (row_index, mut row) in float_data.row_iter_mut().enumerate() {
        let row_scale = scale[row_index];
        for val in row.values_mut() {
            *val *= row_scale;
        }
    }

    drop(read_guard);
    let _ = x_data.0.insert(ArrayData::from(float_data));
    Ok(())
}

pub(crate) fn scale_col(adata: &mut IMAnnData, target_value: f64) -> anyhow::Result<()> {
    let sum_col = crate::memory::statistics::compute_sum(adata, crate::Direction::Column)?;
    let non_zero_col = crate::memory::statistics::compute_number(adata, crate::Direction::Column)?;
    let scale = sum_col.iter().zip(non_zero_col.iter()).map(|(sum, non_zero)| {
        if *non_zero == 0 {
            0.0
        } else {
            target_value / (*sum / *non_zero as f64)
        }
    }).collect::<Vec<f64>>();

    let x = adata.x();
    match x.get_type()? {
        anndata::backend::DataType::Array(_) => todo!(),
        anndata::backend::DataType::Categorical => todo!(),
        anndata::backend::DataType::CsrMatrix(_) => scale_col_csr(adata, scale),
        anndata::backend::DataType::CscMatrix(_) => scale_col_csc(adata, scale),
        anndata::backend::DataType::DataFrame => todo!(),
        anndata::backend::DataType::Scalar(_) => todo!(),
        anndata::backend::DataType::Mapping => todo!(),
    }
}

fn scale_col_csc(adata: &mut IMAnnData, scale: Vec<f64>) -> anyhow::Result<()> {
    let x_data = adata.x();
    let read_guard = x_data.0.read_inner();
    let arr_data_clone = read_guard.clone();
    let mut float_data: CscMatrix<f64> = arr_data_clone.try_into().expect("Failed to convert to CscMatrix<f64>");
    let (col_offsets, _row_indices, values) = float_data.csc_data_mut();
    for col in 0..col_offsets.len() - 1 {
        for j in col_offsets[col]..col_offsets[col + 1] {
            values[j] *= scale[col];
        }
    }
    drop(read_guard);
    let _ = x_data.0.insert(ArrayData::from(float_data));
    Ok(())
}

fn scale_col_csr(adata: &mut IMAnnData, scale: Vec<f64>) -> anyhow::Result<()> {
    let x_data = adata.x();
    let read_guard = x_data.0.read_inner();
    let arr_data_clone = read_guard.clone();
    let mut float_data: CsrMatrix<f64> = arr_data_clone.try_into().expect("Failed to convert to CsrMatrix<f64>");
    let (row_offsets, col_indices, values) = float_data.csr_data_mut();
    for row in 0..row_offsets.len() - 1 {
        for j in row_offsets[row]..row_offsets[row + 1] {
            let col = col_indices[j];
            values[j] *= scale[col];
        }
    }
    drop(read_guard);
    let _ = x_data.0.insert(ArrayData::from(float_data));
    Ok(())
}
