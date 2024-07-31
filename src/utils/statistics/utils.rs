use anndata::{
    data::{DynCscMatrix, DynCsrMatrix},
    AnnData, AnnDataOp, ArrayData, ArrayElem, ArrayElemOp, Backend,
};
use anyhow::Ok;
use nalgebra_sparse::{csr::CsrMatrix, CscMatrix};
mod csc;
mod csr;

pub fn compute_n_genes<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_genes non-chunked");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    log::debug!("Loaded array data.");
    let n_genes = match x {
        ArrayData::CscMatrix(csc) => csc::compute_n_genes(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_n_genes(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    };

    n_genes
}

pub fn compute_n_genes_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_genes chunked {}", chunk_size);
    let x = adata.x();
    log::debug!("Loaded X into memory");
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_rows = shape[0];
    log::debug!("Loaded shape successfully, loading array data!");
    let arr_data = x.get::<ArrayData>()?;
    log::debug!("Loaded array data, doing match!");
    match arr_data {
        Some(ArrayData::CscMatrix(_)) => csc::chunked_n_genes(&x, n_rows, chunk_size),
        Some(ArrayData::CsrMatrix(_)) => csr::chunked_n_genes(&x, n_rows, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_n_cells<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_cells non-chunked");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    log::debug!("Loaded array data.");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_n_cells(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_n_cells(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_n_cells_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_cells chunked {}", chunk_size);
    let x = adata.x();
    log::debug!("Loaded X into memory");
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    log::debug!("Loaded shape successfully, loading array data!");
    let arr_data = x.get::<ArrayData>()?;
    log::debug!("Loaded array data, doing match!");
    match arr_data {
        Some(ArrayData::CscMatrix(_)) => csc::chunked_n_cells(&x, n_cols, chunk_size),
        Some(ArrayData::CsrMatrix(_)) => csr::chunked_n_cells(&x, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_n_combined<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_cells non-chunked");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    let n_rows = shape[0];
    log::debug!("Loaded array data.");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_n_combined(&csc, n_rows, n_cols),
        ArrayData::CsrMatrix(csr) => csr::compute_n_combined(&csr, n_rows, n_cols),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_n_combined_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_cells chunked {}", chunk_size);
    let x = adata.x();
    log::debug!("Loaded X into memory");
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    let n_rows = shape[0];
    log::debug!("Loaded shape successfully, loading array data!");
    let arr_data = x.get::<ArrayData>()?;
    log::debug!("Loaded array data, doing match!");
    match arr_data {
        Some(ArrayData::CscMatrix(_)) => csc::chunked_n_cells(&x, n_cols, chunk_size),
        Some(ArrayData::CsrMatrix(_)) => csr::chunked_n_cells(&x, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}
