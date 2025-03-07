use polars::prelude::{IntoColumn, NamedFrom, NamedFromOwned};
use polars::series::Series;
use std::collections::HashMap;
use std::ops::DerefMut;

use anndata::{backend::DataType, data::DynArray, ArrayData};
use anndata_memory::IMArrayElement;
use anyhow::bail;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use ndarray::{Array, Array2, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use num_traits::{Float, NumCast};
use polars::prelude::{Column, DataFrame};
use single_algebra::NumericOps;
use crate::shared::{need_conversion_target_float_type, Precision};

pub fn target_type_float_need_conversion_in_memory(
    matrix_datatype: &DataType,
) -> anyhow::Result<bool> {
    match matrix_datatype {
        anndata::backend::DataType::Array(scalar_type) => {
            need_conversion_target_float_type(scalar_type)
        }
        anndata::backend::DataType::CsrMatrix(scalar_type) => {
            need_conversion_target_float_type(scalar_type)
        }
        anndata::backend::DataType::CscMatrix(scalar_type) => {
            need_conversion_target_float_type(scalar_type)
        }
        anndata::backend::DataType::DataFrame => {
            bail!("Cannot use a matrix of type <DataFrame> in the normalization procedure.")
        }
        anndata::backend::DataType::Mapping => {
            bail!("Cannot use a matrix of type <Mapping> in the normalization procedure.")
        }
        anndata::backend::DataType::Scalar(scalar_type) => {
            need_conversion_target_float_type(scalar_type)
        }
        anndata::backend::DataType::Categorical => {
            bail!("Cannot use a matrix of type <Categorical> in the normalization procedure.")
        }
        anndata::backend::DataType::NullableArray => {
            bail!("Cannot use a matrix of type <NullableArray> in the normalization procedure.")
        }
    }
}

pub fn convert_to_float_if_non_float_type(
    matrix: &IMArrayElement,
    precision: Option<Precision>,
) -> anyhow::Result<()> {
    let matrix_data_type = matrix.get_type()?;
    let precision = match precision {
        Some(prec) => prec,
        None => Precision::default(),
    };

    // For now discarded, as we want to convert f32 -> f64 and f64 -> f32 in case this becomes necessary
    //let need_to_convert_type = target_type_float_need_conversion_in_memory(&matrix_data_type)?;

    //if !need_to_convert_type {
    //    return Ok(());
    //}

    let mut write_guard = matrix.0.write_inner();
    let mut data = write_guard.deref_mut();

    let dummy_data: Array2<f64> = Array2::zeros((0, 0));
    let dummy_array_data = ArrayData::Array(DynArray::from(dummy_data));
    let original_matrix_data = std::mem::replace(data, dummy_array_data);

    let new_matrix: anyhow::Result<ArrayData> = match original_matrix_data {
        ArrayData::Array(dyn_array) => {
            match (dyn_array, precision) {
                (DynArray::I8(array_base), Precision::Single) => {
                    let converted = convert_array::<i8, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I8(array_base), Precision::Double) => {
                    let converted = convert_array::<i8, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I16(array_base), Precision::Single) => {
                    let converted = convert_array::<i16, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I16(array_base), Precision::Double) => {
                    let converted = convert_array::<i16, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I32(array_base), Precision::Single) => {
                    let converted = convert_array::<i32, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I32(array_base), Precision::Double) => {
                    let converted = convert_array::<i32, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I64(array_base), Precision::Single) => {
                    let converted = convert_array::<i64, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::I64(array_base), Precision::Double) => {
                    let converted = convert_array::<i64, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U8(array_base), Precision::Single) => {
                    let converted = convert_array::<u8, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U8(array_base), Precision::Double) => {
                    let converted = convert_array::<u8, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U16(array_base), Precision::Single) => {
                    let converted = convert_array::<u16, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U16(array_base), Precision::Double) => {
                    let converted = convert_array::<u16, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U32(array_base), Precision::Single) => {
                    let converted = convert_array::<u32, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U32(array_base), Precision::Double) => {
                    let converted = convert_array::<u32, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U64(array_base), Precision::Single) => {
                    let converted = convert_array::<u64, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::U64(array_base), Precision::Double) => {
                    let converted = convert_array::<u64, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::F32(array_base), Precision::Single) => Ok(ArrayData::from(array_base)),
                (DynArray::F32(array_base), Precision::Double) => {
                    let converted = convert_array::<f32, f64>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::F64(array_base), Precision::Single) => {
                    let converted = convert_array::<f64, f32>(array_base)?;
                    Ok(ArrayData::from(converted))
                },
                (DynArray::F64(array_base), Precision::Double) => Ok(ArrayData::from(array_base)),
                (DynArray::Bool(array_base), Precision::Single) => bail!("ArrayBase with type: <bool> cannot be converted into float<f32>. Please convert it manually before."),
                (DynArray::Bool(array_base), Precision::Double) => bail!("ArrayBase with type: <bool> cannot be converted into float<f64>. Please convert it manually before."),
                (DynArray::String(array_base), Precision::Single) => bail!("ArrayBase with type: <string> cannot be converted into float<f32>. Please convert it manually before."),
                (DynArray::String(array_base), Precision::Double) => bail!("ArrayBase with type: <string> cannot be converted into float<f64>. Please convert it manually before."),
            }
        },
        ArrayData::CsrMatrix(dyn_csr_matrix) => match (dyn_csr_matrix, precision) {
            (anndata::data::DynCsrMatrix::I8(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<i8, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I8(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<i8, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I16(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<i16, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I16(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<i16, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I32(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<i32, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I32(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<i32, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I64(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<i64, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::I64(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<i64, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U8(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<u8, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U8(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<u8, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U16(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<u16, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U16(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<u16, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U32(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<u32, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U32(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<u32, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U64(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<u64, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::U64(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<u64, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::F32(csr_matrix), Precision::Single) => Ok(ArrayData::from(csr_matrix)),
            (anndata::data::DynCsrMatrix::F32(csr_matrix), Precision::Double) => {
                let converted = convert_csr_sparse_matrix::<f32, f64>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::F64(csr_matrix), Precision::Single) => {
                let converted = convert_csr_sparse_matrix::<f64, f32>(csr_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCsrMatrix::F64(csr_matrix), Precision::Double) => Ok(ArrayData::from(csr_matrix)),
            (anndata::data::DynCsrMatrix::Bool(_), Precision::Single) => bail!("CsrMatrix with type: <bool> cannot be converted into float<f32>. Please convert it manually before."),
            (anndata::data::DynCsrMatrix::Bool(_), Precision::Double) => bail!("CsrMatrix with type: <bool> cannot be converted into float<f64>. Please convert it manually before."),
            (anndata::data::DynCsrMatrix::String(_), Precision::Single) => bail!("CsrMatrix with type: <string> cannot be converted into float<f32>. Please convert it manually before."),
            (anndata::data::DynCsrMatrix::String(_), Precision::Double) => bail!("CsrMatrix with type: <string> cannot be converted into float<f64>. Please convert it manually before."),
        },
        ArrayData::CsrNonCanonical(_) => todo!("This is not implemented yet!"),
        ArrayData::CscMatrix(dyn_csc_matrix) => match (dyn_csc_matrix, precision) {
            (anndata::data::DynCscMatrix::I8(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<i8, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I8(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<i8, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I16(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<i16, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I16(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<i16, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I32(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<i32, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I32(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<i32, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I64(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<i64, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::I64(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<i64, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U8(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<u8, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U8(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<u8, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U16(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<u16, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U16(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<u16, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U32(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<u32, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U32(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<u32, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U64(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<u64, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::U64(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<u64, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::F32(csc_matrix), Precision::Single) => Ok(ArrayData::from(csc_matrix)),
            (anndata::data::DynCscMatrix::F32(csc_matrix), Precision::Double) => {
                let converted = convert_csc_sparse_matrix::<f32, f64>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::F64(csc_matrix), Precision::Single) => {
                let converted = convert_csc_sparse_matrix::<f64, f32>(csc_matrix)?;
                Ok(ArrayData::from(converted))
            },
            (anndata::data::DynCscMatrix::F64(csc_matrix), Precision::Double) => Ok(ArrayData::from(csc_matrix)),
            (anndata::data::DynCscMatrix::Bool(_), Precision::Single) => bail!("CscMatrix with type: <bool> cannot be converted into float<f32>. Please convert it manually before."),
            (anndata::data::DynCscMatrix::Bool(_), Precision::Double) => bail!("CscMatrix with type: <bool> cannot be converted into float<f64>. Please convert it manually before."),
            (anndata::data::DynCscMatrix::String(_), Precision::Single) => bail!("CscMatrix with type: <string> cannot be converted into float<f32>. Please convert it manually before."),
            (anndata::data::DynCscMatrix::String(_), Precision::Double) => bail!("CscMatrix with type: <string> cannot be converted into float<f32>. Please convert it manually before."),
        },
        ArrayData::DataFrame(_) => todo!("Conversion with dataframes has not been implemented yet!"),
    };

    *data = new_matrix?;

    Ok(())
}

fn convert_csr_sparse_matrix<T, U>(matrix: CsrMatrix<T>) -> anyhow::Result<CsrMatrix<U>>
where
    T: NumericOps + NumCast + Copy, // Base numeric traits for source
    U: NumericOps + NumCast + Copy + Float, // Ensure target is float (f32/f64)
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let (row_offsets, col_indices, values) = matrix.disassemble();

    let new_values: Vec<U> = values
        .into_iter()
        .map(|x| NumCast::from(x).unwrap())
        .collect();

    CsrMatrix::try_from_csr_data(nrows, ncols, row_offsets, col_indices, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {}", e))
}

fn convert_csc_sparse_matrix<T, U>(matrix: CscMatrix<T>) -> anyhow::Result<CscMatrix<U>>
where
    T: NumericOps + NumCast + Copy, // Base numeric traits for source
    U: NumericOps + NumCast + Copy + Float, // Ensure target is float (f32/f64)
{
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let (col_offsets, row_indices, values) = matrix.disassemble();

    let new_values: Vec<U> = values
        .into_iter()
        .map(|x| NumCast::from(x).unwrap())
        .collect();

    CscMatrix::try_from_csc_data(nrows, ncols, col_offsets, row_indices, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSC matrix: {}", e))
}

fn convert_array<T, U>(
    array: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>,
) -> anyhow::Result<ArrayBase<OwnedRepr<U>, Dim<IxDynImpl>>>
where
    T: NumericOps + NumCast + Copy,
    U: NumericOps + NumCast + Copy + Float,
{
    let shape = array.raw_dim(); // Keep the dynamic dimension

    let (vec, _) = array.into_raw_vec_and_offset();

    let new_values: Vec<U> = vec.into_iter().map(|x| NumCast::from(x).unwrap()).collect();

    Ok(ArrayBase::from_shape_vec(shape, new_values)?)
}

pub fn create_dataframe_from_map<T>(
    map: &HashMap<String, Vec<T>>
) -> anyhow::Result<DataFrame>
where
    T: Clone, polars::prelude::Series: polars::prelude::NamedFromOwned<std::vec::Vec<T>> {
    let mut df = DataFrame::default();

    for (group, values) in map {
        let ser = polars::prelude::Series::from_vec(group.into(), values.clone()).into_column();
        df.with_column(ser)?;
    }
    Ok(df)
}

pub fn create_string_dataframe_from_map(
    map: &HashMap<String, Vec<String>>
) -> anyhow::Result<DataFrame> {
    let mut df = DataFrame::default();

    for (group, values) in map {
        let string_slice: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let series = Series::new(group.into(), &string_slice);
        df.with_column(series)?;
    }

    Ok(df)
}

