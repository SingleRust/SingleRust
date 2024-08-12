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