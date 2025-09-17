//! # Memory Utilities for Single-Cell Data Processing
//!
//! This module provides essential utility functions for handling data type conversions,
//! matrix transformations, and data structure manipulations in single-cell analysis.
//! These utilities are crucial for ensuring compatibility between different numeric types
//! and data formats used throughout the SingleRust ecosystem.
//!
//! ## Key Functionality
//!
//! ### Type Conversions
//! - **Float conversions**: Convert integer/other numeric types to f32/f64 for analysis
//! - **Precision control**: Handle single vs double precision requirements
//! - **Matrix format preservation**: Maintain sparse matrix structure during conversion
//!
//! ### Data Structure Utilities
//! - **DataFrame creation**: Convert analysis results to Polars DataFrames
//! - **Array conversions**: Transform between different ndarray types
//! - **Result formatting**: Prepare data for storage in AnnData objects
//!
//! ## Why Type Conversion Matters
//!
//! Single-cell data often comes in various numeric formats:
//! - **Raw counts**: Usually integers (u32, i32)
//! - **Normalized data**: Requires floating point (f32, f64)
//! - **Statistical results**: Often need double precision (f64)
//! - **Memory efficiency**: Sometimes f32 is preferred for large datasets
//!
//! This module handles these conversions safely and efficiently while preserving
//! data structure characteristics like sparsity.
//!
//! ## Usage Patterns
//!
//! ```rust,ignore
//! use single_rust::memory::utils::{
//!     convert_to_float_if_non_float_type,
//!     create_dataframe_from_map,
//!     arr1_conversion
//! };
//! use single_rust::shared::Precision;
//!
//! // Convert matrix to double precision for analysis
//! convert_to_float_if_non_float_type(&adata.x(), Some(Precision::Double))?;
//!
//! // Create DataFrame from analysis results
//! let mut results = HashMap::new();
//! results.insert("group_A".to_string(), vec![1.0, 2.0, 3.0]);
//! results.insert("group_B".to_string(), vec![4.0, 5.0, 6.0]);
//! let df = create_dataframe_from_map(&results)?;
//! ```
//!
//! ## Supported Conversions
//!
//! ### Matrix Types
//! - **Dense arrays**: ndarray Array types
//! - **CSR sparse matrices**: Compressed Sparse Row format
//! - **CSC sparse matrices**: Compressed Sparse Column format
//!
//! ### Numeric Types
//! - **Integers**: i8, i16, i32, i64, u8, u16, u32, u64
//! - **Floats**: f32, f64
//! - **Target precision**: Single (f32) or Double (f64)
//!
//! ## Performance Considerations
//!
//! - **In-place conversion**: Minimizes memory usage by modifying existing data
//! - **Sparse preservation**: Maintains sparsity structure during conversion
//! - **Batch operations**: Efficient vectorized conversions
//! - **Zero-copy when possible**: Avoids unnecessary data copying

use polars::prelude::{IntoColumn, NamedFrom, NamedFromOwned};
use polars::series::Series;
use std::collections::HashMap;
use std::ops::DerefMut;

use crate::shared::{need_conversion_target_float_type, Precision};
use anndata::{backend::DataType, data::DynArray, ArrayData};
use anndata_memory::IMArrayElement;
use anyhow::{anyhow, bail};
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use ndarray::{Array1, Array2, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use num_traits::{Float, Num, NumCast};
use polars::prelude::DataFrame;
use single_utilities::traits::NumericOps;

/// Check if a matrix data type requires conversion to float for processing.
///
/// Determines whether the given matrix data type needs to be converted to a floating-point
/// type for mathematical operations. Most single-cell analysis requires floating-point
/// arithmetic for normalization, statistical tests, and dimensionality reduction.
///
/// ## Parameters
/// * `matrix_datatype` - The AnnData backend data type to check
///
/// ## Returns
/// * `Ok(true)` - Conversion to float is needed
/// * `Ok(false)` - Data is already in appropriate float format
/// * `Err` - Unsupported data type for conversion
///
/// ## Supported Types
/// - Array types (dense matrices)
/// - CSR/CSC sparse matrices  
/// - Scalar values
///
/// ## Unsupported Types
/// - DataFrames (use specialized handling)
/// - Mappings (not suitable for numeric operations)
/// - Categorical data (convert manually first)
/// - NullableArrays (handle nulls before conversion)
pub fn _target_type_float_need_conversion_in_memory(
    matrix_datatype: &DataType,
) -> anyhow::Result<bool> {
    match matrix_datatype {
        DataType::Array(scalar_type) => need_conversion_target_float_type(scalar_type),
        DataType::CsrMatrix(scalar_type) => need_conversion_target_float_type(scalar_type),
        DataType::CscMatrix(scalar_type) => need_conversion_target_float_type(scalar_type),
        DataType::DataFrame => {
            bail!("Cannot use a matrix of type <DataFrame> in the normalization procedure.")
        }
        DataType::Mapping => {
            bail!("Cannot use a matrix of type <Mapping> in the normalization procedure.")
        }
        DataType::Scalar(scalar_type) => need_conversion_target_float_type(scalar_type),
        DataType::Categorical => {
            bail!("Cannot use a matrix of type <Categorical> in the normalization procedure.")
        }
        DataType::NullableArray => {
            bail!("Cannot use a matrix of type <NullableArray> in the normalization procedure.")
        }
    }
}

/// Convert matrix data to floating-point format if needed for analysis.
///
/// This is the main conversion function that handles in-place transformation of matrix data
/// from integer or other numeric types to floating-point types (f32 or f64). Essential for
/// most single-cell analysis operations that require floating-point arithmetic.
///
/// ## Key Features
/// - **In-place conversion**: Modifies existing data to minimize memory usage
/// - **Precision control**: Choose between single (f32) or double (f64) precision
/// - **Sparse preservation**: Maintains sparse matrix structure during conversion
/// - **Type safety**: Comprehensive error handling for unsupported conversions
///
/// ## Parameters
/// * `matrix` - The matrix element to convert (modified in-place)
/// * `precision` - Target precision (Single=f32, Double=f64, None=Double)
///
/// ## Supported Source Types
/// - **Integers**: i8, i16, i32, i64, u8, u16, u32, u64
/// - **Floats**: f32 ↔ f64 conversion
/// - **Matrix formats**: Dense arrays, CSR matrices, CSC matrices
///
/// ## When to Use
/// - Before normalization operations
/// - Prior to statistical analysis
/// - When switching between f32/f64 for memory vs precision trade-offs
/// - Before applying mathematical transformations
///
/// ## Examples
/// ```rust,ignore
/// // Convert to double precision for high-accuracy analysis
/// convert_to_float_if_non_float_type(&adata.x(), Some(Precision::Double))?;
///
/// // Convert to single precision to save memory
/// convert_to_float_if_non_float_type(&adata.x(), Some(Precision::Single))?;
///
/// // Use default (double precision)
/// convert_to_float_if_non_float_type(&adata.x(), None)?;
/// ```
///
/// ## Performance Notes
/// - In-place operation minimizes memory overhead
/// - Vectorized conversions for efficiency
/// - Preserves sparse matrix structure (no densification)
/// - Zero-copy when source and target types match
pub fn convert_to_float_if_non_float_type(
    matrix: &IMArrayElement,
    precision: Option<Precision>,
) -> anyhow::Result<()> {
    let precision = precision.unwrap_or_default();

    // For now discarded, as we want to convert f32 -> f64 and f64 -> f32 in case this becomes necessary
    //let need_to_convert_type = target_type_float_need_conversion_in_memory(&matrix_data_type)?;

    //if !need_to_convert_type {
    //    return Ok(());
    //}

    let mut write_guard = matrix.0.write_inner();
    let data = write_guard.deref_mut();

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
                (DynArray::Bool(_), Precision::Single) => bail!("ArrayBase with type: <bool> cannot be converted into float<f32>. Please convert it manually before."),
                (DynArray::Bool(_), Precision::Double) => bail!("ArrayBase with type: <bool> cannot be converted into float<f64>. Please convert it manually before."),
                (DynArray::String(_), Precision::Single) => bail!("ArrayBase with type: <string> cannot be converted into float<f32>. Please convert it manually before."),
                (DynArray::String(_), Precision::Double) => bail!("ArrayBase with type: <string> cannot be converted into float<f64>. Please convert it manually before."),
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

/// Convert CSR sparse matrix between numeric types.
///
/// Transforms a Compressed Sparse Row matrix from one numeric type to another while
/// preserving the sparse structure. Particularly useful for converting integer count
/// data to floating-point for analysis.
///
/// ## Parameters
/// * `matrix` - Source CSR matrix to convert
///
/// ## Type Parameters
/// * `T` - Source numeric type (e.g., i32, u32)
/// * `U` - Target floating-point type (f32 or f64)
///
/// ## Returns
/// New CSR matrix with converted values in target type
///
/// ## Preservation
/// - **Sparsity pattern**: Row offsets and column indices unchanged
/// - **Matrix dimensions**: Rows and columns preserved
/// - **Memory efficiency**: No unnecessary densification
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

/// Convert CSC sparse matrix between numeric types.
///
/// Transforms a Compressed Sparse Column matrix from one numeric type to another while
/// preserving the sparse structure. Similar to CSR conversion but for column-major format.
///
/// ## Parameters
/// * `matrix` - Source CSC matrix to convert
///
/// ## Type Parameters  
/// * `T` - Source numeric type (e.g., i32, u32)
/// * `U` - Target floating-point type (f32 or f64)
///
/// ## Returns
/// New CSC matrix with converted values in target type
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

/// Convert dense ndarray between numeric types.
///
/// Transforms a dense array from one numeric type to another, preserving the
/// multidimensional structure. Handles dynamic dimensionality efficiently.
///
/// ## Parameters
/// * `array` - Source array with dynamic dimensions
///
/// ## Type Parameters
/// * `T` - Source numeric type
/// * `U` - Target floating-point type
///
/// ## Returns
/// New array with same shape but converted element type
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

/// Create a Polars DataFrame from a HashMap of numeric vectors.
///
/// Converts analysis results stored as HashMap<String, Vec<T>> into a structured
/// DataFrame format suitable for storage in AnnData objects or further analysis.
/// Commonly used for differential expression results, QC metrics, and other
/// group-wise analysis outputs.
///
/// ## Parameters
/// * `map` - HashMap where keys are column names and values are data vectors
///
/// ## Type Parameters
/// * `T` - Numeric type that can be stored in Polars Series
///
/// ## Returns
/// Polars DataFrame with columns corresponding to HashMap keys
///
/// ## Usage Examples
/// ```rust,ignore
/// let mut results = HashMap::new();
/// results.insert("group_A_scores".to_string(), vec![1.5, 2.3, 0.8]);
/// results.insert("group_B_scores".to_string(), vec![0.2, 1.9, 2.1]);
///
/// let df = create_dataframe_from_map(&results)?;
/// // Results in DataFrame with columns "group_A_scores" and "group_B_scores"
/// ```
///
/// ## Common Use Cases
/// - Storing differential expression results by group
/// - Organizing QC metrics by sample/condition
/// - Converting analysis outputs for AnnData storage
/// - Preparing data for visualization or export
pub fn create_dataframe_from_map<T>(map: &HashMap<String, Vec<T>>) -> anyhow::Result<DataFrame>
where
    T: Clone,
    Series: NamedFromOwned<Vec<T>>,
{
    let mut df = DataFrame::default();

    for (group, values) in map {
        let ser = polars::prelude::Series::from_vec(group.into(), values.clone()).into_column();
        df.with_column(ser)?;
    }
    Ok(df)
}

/// Create a Polars DataFrame from a HashMap of String vectors.
///
/// Specialized version of DataFrame creation for string data, commonly used for
/// gene names, cell identifiers, and categorical analysis results.
///
/// ## Parameters
/// * `map` - HashMap where keys are column names and values are String vectors
///
/// ## Returns
/// Polars DataFrame with string columns
///
/// ## Usage Examples
/// ```rust,ignore
/// let mut gene_results = HashMap::new();
/// gene_results.insert("group_A_genes".to_string(),
///                    vec!["ACTB".to_string(), "GAPDH".to_string()]);
/// gene_results.insert("group_B_genes".to_string(),
///                    vec!["TP53".to_string(), "MYC".to_string()]);
///
/// let df = create_string_dataframe_from_map(&gene_results)?;
/// ```
///
/// ## Common Use Cases
/// - Storing gene names from differential expression analysis
/// - Organizing cell type annotations by group
/// - Creating lookup tables for identifiers
/// - Preparing categorical results for storage
pub fn create_string_dataframe_from_map(
    map: &HashMap<String, Vec<String>>,
) -> anyhow::Result<DataFrame> {
    let mut df = DataFrame::default();

    for (group, values) in map {
        let string_slice: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let series = Series::new(group.into(), &string_slice);
        df.with_column(series)?;
    }

    Ok(df)
}

/// Convert a 2D array between numeric types.
///
/// Type-safe conversion between different numeric types for 2D arrays, commonly
/// used when interfacing between different libraries or precision requirements.
///
/// ## Parameters
/// * `array2` - Source 2D array to convert
///
/// ## Type Parameters
/// * `M` - Source numeric type
/// * `T` - Target numeric type
///
/// ## Returns
/// New 2D array with same dimensions but converted element type
///
/// ## Usage
/// ```rust,ignore
/// let f32_array: Array2<f32> = Array2::zeros((10, 5));
/// let f64_array: Array2<f64> = arr2_conversion(f32_array)?;
/// ```
///
/// ## Common Scenarios
/// - Converting PCA results between precision levels
/// - Interfacing with external libraries requiring specific types
/// - Preparing data for storage or analysis pipelines
pub fn arr2_conversion<M, T>(array2: Array2<M>) -> anyhow::Result<Array2<T>>
where
    M: Num + Copy + num_traits::ToPrimitive,
    T: Num + NumCast + Clone,
{
    let mut result = Array2::zeros(array2.dim());

    for (target, &source) in result.iter_mut().zip(array2.iter()) {
        *target = T::from(source).ok_or_else(|| anyhow!("Failed to convert value"))?;
    }

    Ok(result)
}

/// Convert a 1D array between numeric types.
///
/// Type-safe conversion for 1D arrays, useful for converting vectors of statistics,
/// scores, or other single-dimensional data between numeric types.
///
/// ## Parameters
/// * `array1` - Source 1D array to convert
///
/// ## Type Parameters
/// * `M` - Source numeric type
/// * `T` - Target numeric type
///
/// ## Returns
/// New 1D array with same length but converted element type
///
/// ## Usage
/// ```rust,ignore
/// let scores_f32: Array1<f32> = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let scores_f64: Array1<f64> = arr1_conversion(scores_f32)?;
/// ```
///
/// ## Common Use Cases
/// - Converting statistical test results between precision levels
/// - Preparing vectors for different analysis functions
/// - Type compatibility between library interfaces
/// - Converting scores or weights for storage
pub fn arr1_conversion<M, T>(array1: Array1<M>) -> anyhow::Result<Array1<T>>
where
    M: Num + Copy + num_traits::ToPrimitive,
    T: Num + NumCast + Clone,
{
    let mut result = Array1::zeros(array1.dim());

    for (target, &source) in result.iter_mut().zip(array1.iter()) {
        *target = T::from(source).ok_or_else(|| anyhow!("Failed to convert value"))?;
    }

    Ok(result)
}
