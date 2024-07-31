use anndata::{
    data::{DynCscMatrix, DynCsrMatrix},
    AnnData, AnnDataOp, ArrayData, ArrayElem, ArrayElemOp, Backend,
};
use anyhow::Ok;
use nalgebra_sparse::{csr::CsrMatrix, CscMatrix};

/// ---------------------------- Number of genes per cell ----------------------------

pub fn chunked_n_genes<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_genes CSCMatrix chunked {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    log::debug!("Loading iter and processing chunks!");
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk of start: {} end: {}", start, end);
        n_genes_process_chunk(&chunk, &mut n_genes)?;
        log::debug!("Done, next chunk!");
    }
    Ok(n_genes)
}

fn n_genes_process_chunk(csc_chunk: &DynCscMatrix, n_genes: &mut Vec<u32>) -> anyhow::Result<()> {
    log::debug!("Converting chunk from DynCscMatrix into CscMatrix<f64>");
    let csc: CscMatrix<f64> = chunk.try_into()?;
    log::debug!("Calculating number of genes!");
    for &row_idx in csc.row_indices() {
        n_genes[row_idx] += 1;
    }
    Ok(())
}

pub fn compute_n_genes(csc: &DynCscMatrix) -> anyhow::Result<Vec<u32>> {
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

/// ---------------------------- Number of cells per gene ----------------------------

pub fn compute_n_cells(csc: &DynCscMatrix) -> anyhow::Result<Vec<u32>> {
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

pub fn chunked_n_cells<B: Backend>(x: &ArrayElem<B>, n_cols: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
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

fn n_cells_process_chunk(csc_chunk: &DynCscMatrix, n_cells: &mut Vec<u32>) -> anyhow::Result<Vec<u32>> {
    og::debug!("Computing n_cells CSCMatrix, loading matrix.");
    let csc_matrix: CscMatrix<f64> = csc_chunk.try_into()?;
    log::debug!("Loaded matrix successfully!");
    let chunk_n_cells = csc_matrix.col_offsets().windows(2).map(|w| (w[1] - w[0]) as u32);
    for (i, count) in chunk_n_cells.enumerate() {
        n_cells[i] = count;
    }
    Ok(())
}

/// ---------------------------- Number of cells per gene AND number of genes per cell ----------------------------

pub fn chunked_n_combined<T: ArrayElem>(
    x: &T,
    n_rows: usize,
    n_cols: usize,
    chunk_size: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk at: {}, {}", start, end);
        n_combined_process_chunk(&chunk, &mut n_genes, &mut n_cells)?;
    }

    Ok((n_genes, n_cells))
}

fn n_combined_process_chunk(csc_chunk: &DynCscMatrix, n_genes: &mut Vec<u32>, n_cells: &mut Vec<u32>) -> anyhow::Result<Vec<u32>> {
    let csc: CscMatrix<f64> = chunk.try_into()?;
        
    for (col, col_vec) in csc.column_iter().enumerate() {
        n_cells[col] = col_vec.nnz() as u32;
        for (row, _) in col_vec.iter() {
            n_genes[row] += 1;
        }
    }
}

pub fn compute_n_combined(
    csc: &DynCscMatrix,
    n_rows: usize,
    n_cols: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (col, col_vec) in csc_matrix.column_iter().enumerate() {
        n_cells[col] = col_vec.nnz() as u32;
        for (row, _) in col_vec.iter() {
            n_genes[row] += 1;
        }
    }

    Ok((n_genes, n_cells))
}

