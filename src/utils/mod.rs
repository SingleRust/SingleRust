pub mod statistics;
pub mod io;


pub enum ComputationMode {
    Chunked(usize),
    Whole,
}