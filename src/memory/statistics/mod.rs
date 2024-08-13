use std::ops::Deref;

use anndata_memory::IMAnnData;

use crate::shared::{statistics::number::whole, Direction};

pub fn compute_number(adata: IMAnnData, direction: Direction) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    whole(data, direction)
}

pub fn compute_sum(adata: IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::sum::whole(data, direction)
}
