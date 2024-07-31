use anndata::{data::DynCsrMatrix, ArrayElem, ArrayElemOp, Backend};
use nalgebra_sparse::CsrMatrix;

// ---------------------------- Number of genes per cell ----------------------------

pub fn compute_n_genes(csr: &DynCsrMatrix) -> anyhow::Result<Vec<u32>> {
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

pub fn chunked_n_genes<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Compute n_genes CSRMatrix chunked {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    log::debug!("Loading iter and process chunks");
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk of start: {} end: {}", start, end);
        n_genes_process_chunk(&chunk, &mut n_genes, start)?;
        log::debug!("Done, next chunk!");
    }
    Ok(n_genes)
}

fn n_genes_process_chunk(csr_chunk: &DynCsrMatrix, n_genes: &mut Vec<u32>, start: usize) -> anyhow::Result<()> {
    log::debug!("Converting chunk from DynCsrMatrix into CsrMatrix<f64>");
    let csr_matrix: CsrMatrix<f64> = csr_chunk.try_into()?;
    log::debug!("Calculating number of genes!");
    let chunk_n_genes = csr_matrix
        .row_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32);
    for (i, count) in chunk_n_genes.enumerate() {
        n_genes[start + i] = count;
    }
    Ok(())
}

// ---------------------------- Number of cells per gene ----------------------------

pub fn compute_n_cells(csr: &DynCsrMatrix) -> anyhow::Result<Vec<u32>> {
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

pub fn chunked_n_cells<B: Backend>(
    x: &ArrayElem<B>,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing n cells CSRMatrix chunked {}", chunk_size);
    let mut n_cells = vec![0; n_cols];
    log::debug!("Loading iter and processing chunks!");
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}, {}", start, end);
        n_cells_process_chunk(&chunk, &mut n_cells)?;
        log::debug!("Done, next chunk!");
    }
    Ok(n_cells)
}

fn n_cells_process_chunk(
    csr_chunk: &DynCsrMatrix,
    n_cells: &mut Vec<u32>,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Converting chunk from DynCsrMatrix into CsrMatrix<f64>");
    let csr_matrix: CsrMatrix<f64> = csr_chunk.try_into()?;
    log::debug!("Calculating number of cells!");
    for &col_idx in csr_matrix.col_indices() {
        n_cells[col_idx] += 1;
    }
    Ok(())
}

// ---------------------------- Number of cells per gene AND number of genes per cell ----------------------------

pub fn chunked_n_combined<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}, {}", start, end);
        n_combined_process_chunk(&chunk, &mut n_genes, &mut n_cells, start)?;
        log::debug!("Done, next chunk!");
    }

    Ok((n_genes, n_cells))
}

fn n_combined_process_chunk(
    csc_chunk: &DynCsrMatrix,
    n_genes: &mut Vec<u32>,
    n_cells: &mut Vec<u32>,
    start: usize
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Converting chunk from DynCsrMatrix into CsrMatrix<f64>");
    let csr: CsrMatrix<f64> = csc_chunk.try_into()?;
    log::debug!("Calculating number of cells!");
    for (row, row_vec) in csr.row_iter().enumerate() {
        n_genes[start + row] = row_vec.nnz() as u32;
        for (col, _) in row_vec.iter() {
            n_cells[col] += 1;
        }
    }
    Ok(())
}

pub fn compute_n_combined(
    csr: &DynCsrMatrix,
    n_rows: usize,
    n_cols: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into()?;
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (row, row_vec) in csr_matrix.row_iter().enumerate() {
        n_genes[row] = row_vec.nnz() as u32;
        for (col, _) in row_vec.iter() {
            n_cells[col] += 1;
        }
    }

    Ok((n_genes, n_cells))
}


// ---------------------------- Sum of genes per cell ----------------------------

