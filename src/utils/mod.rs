pub mod io;


pub enum ComputationMode {
    Chunked(usize),
    Whole,
}

impl Clone for ComputationMode {
    fn clone(&self) -> Self {
        match self {
            Self::Chunked(arg0) => Self::Chunked(arg0.clone()),
            Self::Whole => Self::Whole,
        }
    }
}