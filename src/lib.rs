//pub mod processing;
//pub mod utils;
//pub mod statistics;

pub mod backed;
pub mod io;
pub mod memory;
pub(crate) mod shared;

pub use shared::ComputationMode;
pub use shared::Direction;
