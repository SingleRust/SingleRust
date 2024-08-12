use anndata_memory::IMAnnData;

use crate::shared::{statistics::number::whole, Direction};

pub fn compute_number(adata: IMAnnData, direction: Direction) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let data = x.get_data()?;
    whole(data, direction)
}

pub fn compute_sum(adata: IMAnnData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    let data = adata.x().get_data()?;
    crate::shared::statistics::sum::whole(data, direction)
}
