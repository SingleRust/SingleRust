pub(crate) mod plot;
pub(crate) mod processing;
pub(crate) mod statistics;
pub(crate) mod utils;

use std::collections::HashMap;
use std::ops::Add;

use anndata::data::{DynArray, DynCscMatrix, DynCsrMatrix, SelectInfoElem};
use anndata::{data::Shape, ArrayData, HasShape};
use anyhow::anyhow;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use ndarray::{Array2, ArrayD, Ix2};
use num_traits::{Bounded, NumCast, One, Zero};
use utils::select_info_elem_to_indices;

pub enum FeatureSelection {
    HighlyVariableCol(String),
    HighlyVariable(usize),
    Randomized(usize),
    VarianceThreshold(f64),
    None,
}

pub enum ComputationMode {
    Chunked(usize),
    Whole,
}

impl Clone for ComputationMode {
    fn clone(&self) -> Self {
        match self {
            Self::Chunked(arg0) => Self::Chunked(*arg0),
            Self::Whole => Self::Whole,
        }
    }
}

pub enum Direction {
    Row = 0,
    Column = 1,
}

impl Clone for Direction {
    fn clone(&self) -> Self {
        match self {
            Self::Row => Self::Row,
            Self::Column => Self::Column,
        }
    }
}

impl Direction {
    pub fn is_row(&self) -> bool {
        match self {
            Self::Row => true,
            Self::Column => false,
        }
    }
}

pub enum FlexValue {
    Absolute(u32),
    Relative(f64),
    None,
}

impl Clone for FlexValue {
    fn clone(&self) -> Self {
        match self {
            Self::Absolute(arg0) => Self::Absolute(*arg0),
            Self::Relative(arg0) => Self::Relative(*arg0),
            Self::None => Self::None,
        }
    }
}

impl FlexValue {
    pub fn is_absolute(&self) -> bool {
        match self {
            Self::Absolute(_) => true,
            Self::Relative(_) => false,
            Self::None => false,
        }
    }

    pub fn is_relative(&self) -> bool {
        match self {
            Self::Absolute(_) => false,
            Self::Relative(_) => true,
            Self::None => false,
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Self::Absolute(_) => false,
            Self::Relative(_) => false,
            Self::None => true,
        }
    }
}

trait NumericOps: Zero + One + NumCast + Copy + std::ops::AddAssign + PartialOrd + Bounded + Add<Output = Self> {}
impl<T: Zero + One + NumCast + Copy + std::ops::AddAssign + PartialOrd + Bounded + Add<Output = Self>> NumericOps for T {}

trait FloatOps: NumericOps + num_traits::Float {}
impl<T: NumericOps + num_traits::Float> FloatOps for T {}

#[macro_export]
macro_rules! match_dyn_csr_matrix {
    ($csr:expr, $fun:ident, $($arg:expr),*) => {
        match $csr {
            DynCsrMatrix::I8(d) => $fun(d, $($arg),*),
            DynCsrMatrix::I16(d) => $fun(d, $($arg),*),
            DynCsrMatrix::I32(d) => $fun(d, $($arg),*),
            DynCsrMatrix::I64(_d) => panic!("I64 CSR matrices are not supported for this operation"),
            DynCsrMatrix::U8(d) => $fun(d, $($arg),*),
            DynCsrMatrix::U16(d) => $fun(d, $($arg),*),
            DynCsrMatrix::U32(d) => $fun(d, $($arg),*),
            DynCsrMatrix::U64(_d) => panic!("U64 CSR matrices are not supported for this operation"),
            DynCsrMatrix::Usize(_d) => panic!("Usize CSR matrices are not supported for this operation"),
            DynCsrMatrix::F32(d) => $fun(d, $($arg),*),
            DynCsrMatrix::F64(d) => $fun(d, $($arg),*),
            DynCsrMatrix::Bool(_) => panic!("Boolean CSR matrices are not supported for this operation"),
            DynCsrMatrix::String(_) => panic!("String CSR matrices are not supported for this operation"),
        }
    };
}

#[macro_export]
macro_rules! match_dyn_csc_matrix {
    ($csc:expr, $fun:ident, $($arg:expr),*) => {
        match $csc {
            DynCscMatrix::I8(d) => $fun(d, $($arg),*),
            DynCscMatrix::I16(d) => $fun(d, $($arg),*),
            DynCscMatrix::I32(d) => $fun(d, $($arg),*),
            DynCscMatrix::I64(_d) => panic!("I64 CSC matrices are not supported for this operation"),
            DynCscMatrix::U8(d) => $fun(d, $($arg),*),
            DynCscMatrix::U16(d) => $fun(d, $($arg),*),
            DynCscMatrix::U32(d) => $fun(d, $($arg),*),
            DynCscMatrix::U64(_d) => panic!("U64 CSC matrices are not supported for this operation"),
            DynCscMatrix::Usize(_d) => panic!("Usize CSC matrices are not supported for this operation"),
            DynCscMatrix::F32(d) => $fun(d, $($arg),*),
            DynCscMatrix::F64(d) => $fun(d, $($arg),*),
            DynCscMatrix::Bool(_) => panic!("Boolean CSC matrices are not supported for this operation"),
            DynCscMatrix::String(_) => panic!("String CSC matrices are not supported for this operation"),
        }
    };
}

pub fn convert_to_array_f64(arr_data: &ArrayData) -> anyhow::Result<Array2<f64>> {
    let shape = arr_data.shape();
    match arr_data {
        ArrayData::Array(array) => convert_to_array_f64_array(array),
        ArrayData::CsrMatrix(csr) => match_dyn_csr_matrix!(csr, convert_to_array_f64_csr, shape),
        ArrayData::CsrNonCanonical(_csc) => todo!(),
        ArrayData::CscMatrix(csc) => match_dyn_csc_matrix!(csc, convert_to_array_f64_csc, shape),
        ArrayData::DataFrame(_) => todo!(),
    }
}

fn convert_to_array_f64_array(darray: &DynArray) -> anyhow::Result<Array2<f64>> {
    match darray {
        DynArray::I8(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::I16(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::I32(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::I64(_) => todo!(),
        DynArray::U8(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::U16(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::U32(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::U64(_) => todo!(),
        DynArray::Usize(_) => todo!(),
        DynArray::F32(arr) => convert_arrayd_to_array2_f64(arr),
        DynArray::F64(array) => convert_arrayd_to_array2_f64(array),
        DynArray::Bool(_) => todo!(),
        DynArray::String(_) => todo!(),
        DynArray::Categorical(_) => todo!(),
    }
}

fn convert_arrayd_to_array2_f64<T: NumericOps>(arrayd: &ArrayD<T>) -> anyhow::Result<Array2<f64>> {
    let shape = arrayd.shape();

    match shape.len() {
        1 => Err(anyhow!("The ArrayD must have at least two dimensions!")),
        2 => Ok(arrayd
            .mapv(|x| NumCast::from(x).unwrap_or_else(f64::zero))
            .into_dimensionality::<Ix2>()?),
        _ => {
            let rows = shape[0];
            let cols = shape[1..].iter().product();
            let flat_data: Vec<f64> = arrayd
                .iter()
                .map(|&x| NumCast::from(x).unwrap_or_else(f64::zero))
                .collect();

            let data = Array2::from_shape_vec((rows, cols), flat_data)?;
            Ok(data)
        }
    }
}

fn convert_to_array_f64_csc<T: NumericOps>(
    csc: &CscMatrix<T>,
    shape: Shape,
) -> anyhow::Result<Array2<f64>> {
    let mut dense = Array2::<f64>::zeros((shape[0], shape[1]));
    for (col, vec) in csc.col_iter().enumerate() {
        for (&row, val) in vec.row_indices().iter().zip(csc.values()) {
            dense[[row, col]] = NumCast::from(*val).unwrap();
        }
    }
    Ok(dense)
}

fn convert_to_array_f64_csr<T: NumericOps>(
    csr: &CsrMatrix<T>,
    shape: Shape,
) -> anyhow::Result<Array2<f64>> {
    let mut dense = Array2::<f64>::zeros((shape[0], shape[1]));
    for (row, vec) in csr.row_iter().enumerate() {
        for (&col, val) in vec.col_indices().iter().zip(csr.values()) {
            dense[[row, col]] = NumCast::from(*val).unwrap();
        }
    }
    Ok(dense)
}

fn convert_to_array_f64_csr_selected<T: NumericOps>(
    csr: &CsrMatrix<T>,
    shape: Shape,
    row_selection: &SelectInfoElem,
    col_selection: &SelectInfoElem,
) -> anyhow::Result<Array2<f64>> {
    let row_indices = select_info_elem_to_indices(row_selection, shape[0])?;
    let col_indices = select_info_elem_to_indices(col_selection, shape[1])?;
    let mut dense = Array2::<f64>::zeros((row_indices.len(), col_indices.len()));

    // Create a mapping from original column indices to output column indices
    let col_map: HashMap<usize, usize> = col_indices.iter().enumerate().map(|(i, &col)| (col, i)).collect();

    for (out_row, &row) in row_indices.iter().enumerate() {
        if row < csr.nrows() {
            let row_start = csr.row_offsets()[row];
            let row_end = csr.row_offsets()[row + 1];
            for (&col, &value) in csr.col_indices()[row_start..row_end]
                .iter()
                .zip(csr.values()[row_start..row_end].iter())
            {
                if let Some(&out_col) = col_map.get(&col) {
                    dense[[out_row, out_col]] = NumCast::from(value)
                        .ok_or_else(|| anyhow!("Failed to convert value to f64"))?;
                }
            }
        }
    }
    Ok(dense)
}

fn convert_to_array_f64_csc_selected<T: NumericOps>(
    csc: &CscMatrix<T>,
    shape: Shape,
    row_selection: &SelectInfoElem,
    col_selection: &SelectInfoElem,
) -> anyhow::Result<Array2<f64>> {
    let row_indices = select_info_elem_to_indices(row_selection, shape[0])?;
    let col_indices = select_info_elem_to_indices(col_selection, shape[1])?;
    let mut dense = Array2::<f64>::zeros((row_indices.len(), col_indices.len()));

    // Create a mapping from original row indices to output row indices
    let row_map: HashMap<usize, usize> = row_indices.iter().enumerate().map(|(i, &row)| (row, i)).collect();

    for (out_col, &col) in col_indices.iter().enumerate() {
        if col < csc.ncols() {
            let col_start = csc.col_offsets()[col];
            let col_end = csc.col_offsets()[col + 1];
            for (&row, &value) in csc.row_indices()[col_start..col_end]
                .iter()
                .zip(csc.values()[col_start..col_end].iter())
            {
                if let Some(&out_row) = row_map.get(&row) {
                    dense[[out_row, out_col]] = NumCast::from(value)
                        .ok_or_else(|| anyhow!("Failed to convert value to f64"))?;
                }
            }
        }
    }
    Ok(dense)
}

pub fn convert_to_array_f64_selected(
    data: &ArrayData,
    shape: Shape,
    row_selection: &SelectInfoElem,
    col_selection: &SelectInfoElem,
) -> anyhow::Result<Array2<f64>> {
    match data {
        ArrayData::CscMatrix(csc) => match_dyn_csc_matrix!(
            csc,
            convert_to_array_f64_csc_selected,
            shape,
            row_selection,
            col_selection
        ),
        ArrayData::CsrMatrix(csr) => match_dyn_csr_matrix!(
            csr,
            convert_to_array_f64_csr_selected,
            shape,
            row_selection,
            col_selection
        ),
        _ => anyhow::bail!("Unsupported data type for conversion to Array2<f64>"),
    }
}

// pub fn convert_to_dense_matrix_f64(arr_data: &ArrayData) -> anyhow::Result<Array2<f64>> {
//     let shape = arr_data.shape();
//     match arr_data {
//         ArrayData::Array(_) => todo!(),
//         ArrayData::CsrMatrix(csr) => match_dyn_csr_matrix!(csr, convert_to_array_f64_csr, shape),
//         ArrayData::CsrNonCanonical(_csc) => todo!(),
//         ArrayData::CscMatrix(csc) => match_dyn_csc_matrix!(csc, convert_to_array_f64_csc, shape),
//         ArrayData::DataFrame(_) => todo!(),
//     }
// }
