//pub mod processing;
//pub mod utils;
//pub mod statistics;

pub mod backed;
pub mod io;
pub mod memory;
pub mod shared;

pub use shared::convert_to_array_f64;
pub use shared::ComputationMode;
pub use shared::FeatureSelection;
pub use shared::FlexValue;

pub use shared::statistics::ComputeMinMax;
pub use shared::statistics::ComputeNonZero;
pub use shared::statistics::ComputeSum;
pub use shared::statistics::ComputeVariance;

