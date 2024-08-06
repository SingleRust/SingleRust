pub mod processing;
pub mod utils;
pub mod io;
pub mod statistics;
mod backend;

//pub use backend::InMemoryAnnData;

#[cfg(test)]
mod tests {
    use anndata::Backend;
    use anndata_hdf5::H5;
    use std::time::Instant;
    use super::*;

}
