use anndata::{AnnData, Backend};

pub enum FilterLimit {
    Absolute(u32),
    Relative(f64),
    None,
}

pub enum NormalizationMethod {
    TotalCount,
    RelativeCount,
}

pub fn remove_cells<B: Backend>(anndata: &mut AnnData<B>) -> anyhow::Result<()> {
    anndata.
}