use anndata::{data::DynCsrMatrix, ArrayElem, ArrayElemOp, Backend};
use nalgebra_sparse::CsrMatrix;

// ---------------------------- Number of genes per cell ----------------------------

pub fn compute_num_genes(csr: &DynCsrMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_genes");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into().expect("Matrix conversion failed");
    log::debug!("Matrix converted");
    let ret = csr_matrix
        .row_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32)
        .collect();
    log::debug!("Data collected");
    Ok(ret)
}

pub fn chunked_num_genes<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Chunked num_genes: size {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_num_genes_chunk(chunk, &mut n_genes, start)?;
    }
    Ok(n_genes)
}

fn process_num_genes_chunk(csr_chunk: DynCsrMatrix, n_genes: &mut Vec<u32>, start: usize) -> anyhow::Result<()> {
    let csr_matrix: CsrMatrix<f64> = csr_chunk.try_into()?;
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

pub fn compute_num_cells(csr: &DynCsrMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_cells");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into()?;
    let mut n_cells = vec![0; csr_matrix.ncols()];
    for &col_idx in csr_matrix.col_indices() {
        n_cells[col_idx] += 1;
    }
    log::debug!("Num_cells computed");
    Ok(n_cells)
}

pub fn chunked_num_cells<B: Backend>(
    x: &ArrayElem<B>,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Chunked num_cells: size {}", chunk_size);
    let mut n_cells = vec![0; n_cols];
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_num_cells_chunk(chunk, &mut n_cells)?;
    }
    Ok(n_cells)
}

fn process_num_cells_chunk(
    csr_chunk: DynCsrMatrix,
    n_cells: &mut Vec<u32>,
) -> anyhow::Result<()> {
    let csr_matrix: CsrMatrix<f64> = csr_chunk.try_into()?;
    for &col_idx in csr_matrix.col_indices() {
        n_cells[col_idx] += 1;
    }
    Ok(())
}

// ---------------------------- Number of cells per gene AND number of genes per cell ----------------------------

pub fn chunked_num_combined<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Chunked num_combined: size {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_num_combined_chunk(chunk, &mut n_genes, &mut n_cells, start)?;
    }

    Ok((n_genes, n_cells))
}

fn process_num_combined_chunk(
    csc_chunk: DynCsrMatrix,
    n_genes: &mut Vec<u32>,
    n_cells: &mut Vec<u32>,
    start: usize
) -> anyhow::Result<()> {
    let csr: CsrMatrix<f64> = csc_chunk.try_into()?;
    for (row, row_vec) in csr.row_iter().enumerate() {
        n_genes[start + row] = row_vec.nnz() as u32;
        for &col in row_vec.col_indices() {
            n_cells[col] += 1;
        }
    }
    Ok(())
}

pub fn compute_num_combined(
    csr: &DynCsrMatrix,
    n_rows: usize,
    n_cols: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Computing num_combined");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into()?;
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (row, row_vec) in csr_matrix.row_iter().enumerate() {
        n_genes[row] = row_vec.nnz() as u32;
        for &col in row_vec.col_indices() {
            n_cells[col] += 1;
        }
    }

    log::debug!("Num_combined computed");
    Ok((n_genes, n_cells))
}

// ---------------------------- Sum of expression values per cell ----------------------------

pub fn compute_sum_cell_expr(csr: &DynCsrMatrix) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_cell_expr");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into()?;
    let ret = csr_matrix.row_iter()
        .map(|row| row.values().iter().sum())
        .collect();
    log::debug!("Sum_cell_expr computed");
    Ok(ret)
}

pub fn chunked_sum_cell_expr<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<f64>> {
    log::debug!("Chunked sum_cell_expr: size {}", chunk_size);
    let mut cell_sums = vec![0.0; n_rows];
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_sum_cell_expr_chunk(chunk, &mut cell_sums, start)?;
    }
    Ok(cell_sums)
}

fn process_sum_cell_expr_chunk(csr_chunk: DynCsrMatrix, cell_sums: &mut Vec<f64>, start: usize) -> anyhow::Result<()> {
    let csr_matrix: CsrMatrix<f64> = csr_chunk.try_into()?;
    for (i, row) in csr_matrix.row_iter().enumerate() {
        cell_sums[start + i] = row.values().iter().sum();
    }
    Ok(())
}

// ---------------------------- Sum of expression values per gene ----------------------------

pub fn compute_sum_gene_expr(csr: &DynCsrMatrix) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_gene_expr");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into()?;
    let mut gene_sums = vec![0.0; csr_matrix.ncols()];
    for (&col_idx, &value) in csr_matrix.col_indices().iter().zip(csr_matrix.values().iter()) {
        gene_sums[col_idx] += value;
    }
    log::debug!("Sum_gene_expr computed");
    Ok(gene_sums)
}

pub fn chunked_sum_gene_expr<T: ArrayElemOp>(
    x: &T,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<f64>> {
    log::debug!("Chunked sum_gene_expr: size {}", chunk_size);
    let mut gene_sums = vec![0.0; n_cols];
    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_sum_gene_expr_chunk(chunk, &mut gene_sums)?;
    }
    Ok(gene_sums)
}

fn process_sum_gene_expr_chunk(
    csr_chunk: DynCsrMatrix,
    gene_sums: &mut Vec<f64>,
) -> anyhow::Result<()> {
    let csr_matrix: CsrMatrix<f64> = csr_chunk.try_into()?;
    for (&col_idx, &value) in csr_matrix.col_indices().iter().zip(csr_matrix.values().iter()) {
        gene_sums[col_idx] += value;
    }
    Ok(())
}

// ---------------------------- Sum of expression values per gene AND per cell ----------------------------

pub fn chunked_sum_combined<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    log::debug!("Chunked sum_combined: size {}", chunk_size);
    let mut cell_sums = vec![0.0; n_rows];
    let mut gene_sums = vec![0.0; n_cols];

    for (chunk, start, end) in x.iter::<DynCsrMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_sum_combined_chunk(chunk, &mut cell_sums, &mut gene_sums, start)?;
    }

    Ok((cell_sums, gene_sums))
}

fn process_sum_combined_chunk(
    csr_chunk: DynCsrMatrix,
    cell_sums: &mut Vec<f64>,
    gene_sums: &mut Vec<f64>,
    start: usize
) -> anyhow::Result<()> {
    let csr: CsrMatrix<f64> = csr_chunk.try_into()?;
    for (row, row_vec) in csr.row_iter().enumerate() {
        let row_sum: f64 = row_vec.values().iter().sum();
        cell_sums[start + row] = row_sum;
        for (&col, &value) in row_vec.col_indices().iter().zip(row_vec.values().iter()) {
            gene_sums[col] += value;
        }
    }
    Ok(())
}

pub fn compute_sum_combined(
    csr: &DynCsrMatrix,
    n_rows: usize,
    n_cols: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    log::debug!("Computing sum_combined");
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into()?;
    let mut cell_sums = vec![0.0; n_rows];
    let mut gene_sums = vec![0.0; n_cols];

    for (row, row_vec) in csr_matrix.row_iter().enumerate() {
        cell_sums[row] = row_vec.values().iter().sum();
        for (&col, &value) in row_vec.col_indices().iter().zip(row_vec.values().iter()) {
            gene_sums[col] += value;
        }
    }

    log::debug!("Sum_combined computed");
    Ok((cell_sums, gene_sums))
}