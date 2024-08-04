pub mod statistics;
pub mod io;
pub mod processing;


pub enum ComputationMode {
    Chunked(usize),
    Whole,
}