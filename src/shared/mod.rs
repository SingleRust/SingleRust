pub(crate) mod processing;
pub(crate) mod utils;
pub(crate) mod statistics;

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
    Column = 1
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

use num_traits::{NumCast, One, Zero};

trait NumericOps: Zero + One + NumCast + Copy + std::ops::AddAssign {}
impl<T: Zero + One + NumCast + Copy + std::ops::AddAssign> NumericOps for T {}


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