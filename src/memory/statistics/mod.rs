use std::ops::Deref;

use anndata_memory::IMAnnData;

use crate::shared::{statistics::number::whole, Direction};

pub fn compute_number(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    whole(data, direction)
}

pub fn compute_sum(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::sum::whole(data, direction)
}

pub fn compute_variance(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::variance::whole(data, direction)
}

pub fn compute_min_max(adata: &IMAnnData, direction: Direction) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::minmax::whole(data, direction)
}

pub fn compute_std_dev(adata: &IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let data_inner = x.0.read_inner();
    let data = data_inner.deref();
    crate::shared::statistics::stddev::whole(data, direction)
}