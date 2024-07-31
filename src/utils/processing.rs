use anndata::{
    data::{DynCscMatrix, DynCsrMatrix}, AnnData, AnnDataOp, ArrayData, ArrayElem, ArrayElemOp, Backend
};
use anyhow::Ok;
use nalgebra_sparse::{csr::CsrMatrix, CscMatrix};

#[cfg(feature = "parallel")]
pub mod parallel;

// GENE SECTION

pub fn compute_n_genes_chunked<B: Backend>(adata: &AnnData<B>, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
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
        Some(ArrayData::CscMatrix(_)) => compute_n_genes_csc_chunked(&x, n_rows, chunk_size),
        Some(ArrayData::CsrMatrix(_)) => compute_n_genes_csr_chunked(&x, n_rows, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_n_genes<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_genes non-chunked");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    log::debug!("Loaded array data.");
    let n_genes = match x {
        ArrayData::CscMatrix(csc) => compute_n_genes_csc(&csc),
        ArrayData::CsrMatrix(csr) => compute_n_genes_csr(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    };

    n_genes
}

fn compute_n_genes_csr(csr: &DynCsrMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_genes CSRMatrix");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into().expect("Could not convert data.");
    log::debug!("Converted matrix into CSRMatrix<f64>");
    let ret = csr_matrix
        .row_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32)
        .collect();
    log::debug!("Collected all data.");
    Ok(ret)
}

fn compute_n_genes_csr_chunked<T: ArrayElemOp>(x: &T, n_rows: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_genes CSRMatrix chunked {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    log::debug!("Loading iter and process chunks");
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk of start: {} end: {}", start, end);
        log::debug!("Loading chunk:");
        let csr: CsrMatrix<f64> = chunk.try_into()?;
        log::debug!("Chunk loaded!");
        let chunk_n_genes = csr.row_offsets().windows(2).map(|w| (w[1] - w[0]) as u32); 
        log::debug!("Chunked n_genes");
        for (i, count) in chunk_n_genes.enumerate() {
            n_genes[start + i] = count;
        }
        log::debug!("Done, next chunk!");
    }
    Ok(n_genes)
}

fn compute_n_genes_csc_chunked<T: ArrayElemOp>(x: &T, n_rows: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_genes CSCMatrix chunked {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    log::debug!("Loading iter and processing chunks!");
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk at position: start: {} finish: {}", start, end);
        log::debug!("Loading matrix for chunk!");
        let csc: CscMatrix<f64> = chunk.try_into()?;
        log::debug!("Loaded matrix for chunk!");
        for &row_idx in csc.row_indices() {
            n_genes[row_idx] += 1;
        }
        log::debug!("Done, next chunk!");
    }
    Ok(n_genes)
}

fn compute_n_genes_csc(csc: &DynCscMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_genes CSCMatrix, loading matrix.");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into().expect("Could not convert data.");
    log::debug!("Loaded matrix successfully!");
    let mut n_genes = vec![0; csc_matrix.nrows()];
    log::debug!("Processing rows!");
    for &row_idx in csc_matrix.row_indices() {
        n_genes[row_idx] += 1;
    }
    log::debug!("Done processing!");
    Ok(n_genes)
}

// CELL SECTION

fn compute_n_cells_csr(csr: &DynCsrMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n cells, loading matrix.");
    let csr_matrix: CsrMatrix<f64> = csr
        .clone()
        .try_into()
        .expect("Could not convert matrix into CSRMatrix<f64>.");
    log::debug!("Loaded matrix successfully and processing rows now!");
    let mut n_cells = vec![0; csr_matrix.ncols()];
    for &col_idx in csr_matrix.col_indices() {
        n_cells[col_idx] += 1;
    }
    log::debug!("Done processing!");
    Ok(n_cells)
}

fn compute_n_cells_csr_chunked<B: Backend>(x: &ArrayElem<B>, n_cols: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n cells CSRMatrix chunked {}", chunk_size);
    let mut n_cells = vec![0; n_cols];
    log::debug!("Loading iter and processing chunks!");
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}, {}", start, end);
        log::debug!("Converting matrix!");
        let converted_chunk: CsrMatrix<f64> = chunk.try_into()?; 
        log::debug!("converted matrix");
        for &col_idx in converted_chunk.col_indices() {
            n_cells[col_idx] += 1;
        }
        log::debug!("Done, next chunk!");
    }
    Ok(n_cells)
}

fn compute_n_cells_csc(csc: &DynCscMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_cells from CSCMatrix, loading matrix...");
    let csc_matrix: CscMatrix<f64> = csc
        .clone()
        .try_into()
        .expect("Could not convert matrix into CscMatrix<f64>.");
    log::debug!("Loaded matrix successfully!");
    let ret = csc_matrix
        .col_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32)
        .collect();
    log::debug!("Collected all data...done!");
    Ok(ret)
}

fn compute_n_cells_csc_chunked<B: Backend>(x: &ArrayElem<B>, n_cols: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_cells CSCMatrix chunked {}", chunk_size);
    let mut n_cells = vec![0; n_cols];
    log::debug!("Loading iter and processing chunks");
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk at: {}, {}", start, end);
        log::debug!("Converting matrix!");
        let csc: CscMatrix<f64> = chunk.try_into()?;
        log::debug!("Converted matrix!");
        let chunk_n_cells = csc.col_offsets().windows(2).map(|w| (w[1] - w[0]) as u32);
        for (i, count) in chunk_n_cells.enumerate() {
            n_cells[i] = count;
        }
        log::debug!("Done, next chunk!");
    }
    Ok(n_cells)
}

pub fn compute_n_cells_chunked<B: Backend>(adata: &AnnData<B>, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_cells Chunked, loading x!");
    let x = adata.x();
    log::debug!("Loaded x from adata");
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    log::debug!("loaded shape!");
    let n_cols = shape[1];
    let dtype = x.inner().dtype();
    match dtype {
        anndata::backend::DataType::CsrMatrix(_) => compute_n_cells_csr_chunked(&x, n_cols, chunk_size),
        anndata::backend::DataType::CscMatrix(_) => compute_n_cells_csc_chunked(&x, n_cols, chunk_size),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X"))
    }
}

pub fn compute_n_cells<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n_cells, loading array data!");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    log::debug!("Loaded array data!");
    match x {
        ArrayData::CscMatrix(csc) => compute_n_cells_csc(&csc),
        ArrayData::CsrMatrix(csr) => compute_n_cells_csr(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}





