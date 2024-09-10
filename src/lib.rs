//pub mod processing;
//pub mod utils;
//pub mod statistics;

pub mod backed;
pub mod io;
pub mod memory;
pub(crate) mod shared;

pub use shared::ComputationMode;
pub use shared::Direction;
pub use shared::FeatureSelection;
pub use shared::FlexValue;
pub use shared::convert_to_array_f64;
pub use shared::plot::PcaPlotSettings;
