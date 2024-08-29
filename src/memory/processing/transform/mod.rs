use std::ops::DerefMut;

use anndata::{data::{DynCscMatrix, DynCsrMatrix}, ArrayData};
use anndata_memory::IMAnnData;
use nalgebra_sparse::{CscMatrix, CsrMatrix};


pub fn log1p_data(adata: &mut IMAnnData) -> anyhow::Result<()> {
    let x_data = adata.x();
    let mut write_guard = x_data.0.write_inner();
    let arr_data = write_guard.deref_mut();

    match arr_data {
        ArrayData::CscMatrix(ref mut csc_matrix) => {
            match csc_matrix {
                DynCscMatrix::F64(matrix) => {
                    for val in matrix.values_mut() {
                        *val = f64::ln_1p(*val);
                    }
                },
                DynCscMatrix::F32(matrix) => {
                    for val in matrix.values_mut() {
                        *val = f32::ln_1p(*val);
                    }
                },
                _ => {
                    // Convert to f64 if it's not already f32 or f64
                    let mut float_matrix: CscMatrix<f64> = csc_matrix.clone().try_into()?;
                    for val in float_matrix.values_mut() {
                        *val = f64::ln_1p(*val);
                    }
                    *csc_matrix = DynCscMatrix::F64(float_matrix);
                }
            }
        }, 
        ArrayData::CsrMatrix(ref mut csr_matrix) => {
            match csr_matrix {
                DynCsrMatrix::F64(matrix) => {
                    for val in matrix.values_mut() {
                        *val = f64::ln_1p(*val);
                    }
                },
                DynCsrMatrix::F32(matrix) => {
                    for val in matrix.values_mut() {
                        *val = f32::ln_1p(*val);
                    }
                },
                _ => {
                    // Convert to f64 if it's not already f32 or f64
                    let mut float_matrix: CsrMatrix<f64> = csr_matrix.clone().try_into()?;
                    for val in float_matrix.values_mut() {
                        *val = f64::ln_1p(*val);
                    }
                    *csr_matrix = DynCsrMatrix::F64(float_matrix);
                }
            }
        },
        _ => return Err(anyhow::anyhow!("X is neither a CSC nor a CSR matrix")),
    }

    Ok(())
}