pub mod processing;
pub mod utils;
pub mod io;
pub mod statistics;
mod backend;

pub use backend::{InMemoryAnnData, AnnotationMatrix, InMemoryElemCollection, InnerElemInMemory};
