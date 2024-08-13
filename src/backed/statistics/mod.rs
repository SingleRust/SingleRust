use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};

use crate::shared::{ComputationMode, Direction};

pub fn compute_number<B: Backend>(
    adata: AnnData<B>,
    direction: Direction,
    mode: ComputationMode,
) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    match mode {
        ComputationMode::Chunked(size) => {
            let length = match direction {
                Direction::Row => adata.n_obs(),
                Direction::Column => adata.n_vars(),
            };
            crate::shared::statistics::number::chunked(&x, size, direction, length)
        }
        ComputationMode::Whole => {
            let array = x.get::<ArrayData>()?.unwrap();
            crate::shared::statistics::number::whole(&array, direction)
        }
    }
}

pub fn compute_sum<B: Backend>(
    adata: AnnData<B>,
    direction: Direction,
    mode: ComputationMode,
) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    match mode {
        ComputationMode::Chunked(size) => {
            let length = match direction {
                Direction::Row => adata.n_obs(),
                Direction::Column => adata.n_vars(),
            };
            crate::shared::statistics::sum::chunked(&x, size, direction, length)
        }
        ComputationMode::Whole => {
            let array = x.get::<ArrayData>()?.unwrap();
            crate::shared::statistics::sum::whole(&array, direction)
        }
    }
}
