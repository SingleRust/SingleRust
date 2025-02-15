use std::ops::DerefMut;

use anndata_memory::IMArrayElement;
use anyhow::bail;
use num_traits::Float;
use single_algebra::Direction;
use single_algebra::Log1P;
use single_algebra::Normalize;

use crate::shared::{statistics::ComputeSum, Precision};

pub fn normalize_expression(
    matrix: &IMArrayElement,
    expression_target: u32,
    direction: &Direction,
    precision: Option<Precision>,
) -> anyhow::Result<()> {
    let precision = precision.unwrap_or_default();

    crate::memory::utils::convert_to_float_if_non_float_type(matrix, Some(precision))?;
    // guarantees that the data is in f32 or f64 format!

    match precision {
        Precision::Single => normalize_with_type::<f32>(matrix, expression_target, direction),
        Precision::Double => normalize_with_type::<f64>(matrix, expression_target, direction),
    }
}

pub fn log1p_expression(
    matrix: &IMArrayElement,
    precision: Option<Precision>,
) -> anyhow::Result<()> {
    let precision = precision.unwrap_or_default();
    crate::memory::utils::convert_to_float_if_non_float_type(matrix, Some(precision))?;
    log1p(matrix)
}

fn log1p(matrix: &IMArrayElement) -> anyhow::Result<()> {
    let mut write_guard = matrix.0.write_inner();
    let data = write_guard.deref_mut();
    match data {
        anndata::ArrayData::Array(dyn_array) => match dyn_array {
            anndata::data::DynArray::I8(_) => {
                bail!("Array - Normalization it not implemented for type <I8>!")
            }
            anndata::data::DynArray::I16(_) => {
                bail!("Array - Normalization it not implemented for type <I16>!")
            }
            anndata::data::DynArray::I32(_) => {
                bail!("Array - Normalization it not implemented for type <I32>!")
            }
            anndata::data::DynArray::I64(_) => {
                bail!("Array - Normalization it not implemented for type <I64>!")
            }
            anndata::data::DynArray::U8(_) => {
                bail!("Array - Normalization it not implemented for type <U8>!")
            }
            anndata::data::DynArray::U16(_) => {
                bail!("Array - Normalization it not implemented for type <U16>!")
            }
            anndata::data::DynArray::U32(_) => {
                bail!("Array - Normalization it not implemented for type <U32>!")
            }
            anndata::data::DynArray::U64(_) => {
                bail!("Array - Normalization it not implemented for type <U64>!")
            }
            anndata::data::DynArray::F32(_) => todo!("Need to implement here!"),
            anndata::data::DynArray::F64(_) => todo!("Need to implement here!"),
            anndata::data::DynArray::Bool(_) => {
                bail!("Array - Normalization it not implemented for type <bool>!")
            }
            anndata::data::DynArray::String(_) => {
                bail!("Array - Normalization it not implemented for type <bool>!")
            }
        },
        anndata::ArrayData::CsrMatrix(dyn_csr_matrix) => match dyn_csr_matrix {
            anndata::data::DynCsrMatrix::I8(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I8>!")
            }
            anndata::data::DynCsrMatrix::I16(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I16>!")
            }
            anndata::data::DynCsrMatrix::I32(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I32>!")
            }
            anndata::data::DynCsrMatrix::I64(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I64>!")
            }
            anndata::data::DynCsrMatrix::U8(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U8>!")
            }
            anndata::data::DynCsrMatrix::U16(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U16>!")
            }
            anndata::data::DynCsrMatrix::U32(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U32>!")
            }
            anndata::data::DynCsrMatrix::U64(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U64>!")
            }
            anndata::data::DynCsrMatrix::F32(csr_matrix) => csr_matrix.log1p_normalize(),
            anndata::data::DynCsrMatrix::F64(csr_matrix) => csr_matrix.log1p_normalize(),
            anndata::data::DynCsrMatrix::Bool(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <bool>!")
            }
            anndata::data::DynCsrMatrix::String(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <string>!")
            }
        },
        anndata::ArrayData::CsrNonCanonical(_) => {
            todo!("This is not implemented yet!")
        }
        anndata::ArrayData::CscMatrix(dyn_csc_matrix) => match dyn_csc_matrix {
            anndata::data::DynCscMatrix::I8(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I8>!")
            }
            anndata::data::DynCscMatrix::I16(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I16>!")
            }
            anndata::data::DynCscMatrix::I32(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I32>!")
            }
            anndata::data::DynCscMatrix::I64(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I64>!")
            }
            anndata::data::DynCscMatrix::U8(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U8>!")
            }
            anndata::data::DynCscMatrix::U16(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U16>!")
            }
            anndata::data::DynCscMatrix::U32(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U32>!")
            }
            anndata::data::DynCscMatrix::U64(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U64>!")
            }
            anndata::data::DynCscMatrix::F32(csc_matrix) => csc_matrix.log1p_normalize(),
            anndata::data::DynCscMatrix::F64(csc_matrix) => csc_matrix.log1p_normalize(),
            anndata::data::DynCscMatrix::Bool(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <bool>!")
            }
            anndata::data::DynCscMatrix::String(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <string>!")
            }
        },
        anndata::ArrayData::DataFrame(_) => todo!("This is not implemented yet!"),
    }
}

fn normalize_with_type<T>(
    matrix: &IMArrayElement,
    expression_target: u32,
    direction: &Direction,
) -> anyhow::Result<()>
where
    T: Float + std::ops::AddAssign + std::iter::Sum + num_traits::NumCast,
{
    let sums: Vec<T> = matrix.sum_whole(direction)?;

    let mut write_guard = matrix.0.write_inner();

    let data = write_guard.deref_mut();

    let target = T::from(expression_target).unwrap();

    match data {
        anndata::ArrayData::Array(dyn_array) => match dyn_array {
            anndata::data::DynArray::I8(_) => {
                bail!("Array - Normalization it not implemented for type <I8>!")
            }
            anndata::data::DynArray::I16(_) => {
                bail!("Array - Normalization it not implemented for type <I16>!")
            }
            anndata::data::DynArray::I32(_) => {
                bail!("Array - Normalization it not implemented for type <I32>!")
            }
            anndata::data::DynArray::I64(_) => {
                bail!("Array - Normalization it not implemented for type <I64>!")
            }
            anndata::data::DynArray::U8(_) => {
                bail!("Array - Normalization it not implemented for type <U8>!")
            }
            anndata::data::DynArray::U16(_) => {
                bail!("Array - Normalization it not implemented for type <U16>!")
            }
            anndata::data::DynArray::U32(_) => {
                bail!("Array - Normalization it not implemented for type <U32>!")
            }
            anndata::data::DynArray::U64(_) => {
                bail!("Array - Normalization it not implemented for type <U64>!")
            }
            anndata::data::DynArray::F32(_) => todo!("Need to implement here!"),
            anndata::data::DynArray::F64(_) => todo!("Need to implement here!"),
            anndata::data::DynArray::Bool(_) => {
                bail!("Array - Normalization it not implemented for type <bool>!")
            }
            anndata::data::DynArray::String(_) => {
                bail!("Array - Normalization it not implemented for type <bool>!")
            }
        },
        anndata::ArrayData::CsrMatrix(dyn_csr_matrix) => match dyn_csr_matrix {
            anndata::data::DynCsrMatrix::I8(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I8>!")
            }
            anndata::data::DynCsrMatrix::I16(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I16>!")
            }
            anndata::data::DynCsrMatrix::I32(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I32>!")
            }
            anndata::data::DynCsrMatrix::I64(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <I64>!")
            }
            anndata::data::DynCsrMatrix::U8(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U8>!")
            }
            anndata::data::DynCsrMatrix::U16(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U16>!")
            }
            anndata::data::DynCsrMatrix::U32(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U32>!")
            }
            anndata::data::DynCsrMatrix::U64(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <U64>!")
            }
            anndata::data::DynCsrMatrix::F32(csr_matrix) => {
                csr_matrix.normalize::<T>(sums.as_slice(), target, direction)
            }
            anndata::data::DynCsrMatrix::F64(csr_matrix) => {
                csr_matrix.normalize::<T>(sums.as_slice(), target, direction)
            }
            anndata::data::DynCsrMatrix::Bool(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <bool>!")
            }
            anndata::data::DynCsrMatrix::String(_) => {
                bail!("CsrMatrix - Normalization it not implemented for type <string>!")
            }
        },
        anndata::ArrayData::CsrNonCanonical(_) => {
            todo!("This is not implemented yet!")
        }
        anndata::ArrayData::CscMatrix(dyn_csc_matrix) => match dyn_csc_matrix {
            anndata::data::DynCscMatrix::I8(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I8>!")
            }
            anndata::data::DynCscMatrix::I16(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I16>!")
            }
            anndata::data::DynCscMatrix::I32(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I32>!")
            }
            anndata::data::DynCscMatrix::I64(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <I64>!")
            }
            anndata::data::DynCscMatrix::U8(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U8>!")
            }
            anndata::data::DynCscMatrix::U16(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U16>!")
            }
            anndata::data::DynCscMatrix::U32(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U32>!")
            }
            anndata::data::DynCscMatrix::U64(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <U64>!")
            }
            anndata::data::DynCscMatrix::F32(csc_matrix) => {
                csc_matrix.normalize::<T>(sums.as_slice(), target, direction)
            }
            anndata::data::DynCscMatrix::F64(csc_matrix) => {
                csc_matrix.normalize::<T>(sums.as_slice(), target, direction)
            }
            anndata::data::DynCscMatrix::Bool(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <bool>!")
            }
            anndata::data::DynCscMatrix::String(_) => {
                bail!("CscMatrix - Normalization it not implemented for type <string>!")
            }
        },
        anndata::ArrayData::DataFrame(_) => todo!("This is not implemented yet!"),
    }
}
