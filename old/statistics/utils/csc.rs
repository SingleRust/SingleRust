use anndata::{
    data::DynCscMatrix, ArrayElem, ArrayElemOp, Backend,
};
use anyhow::Ok;
use nalgebra_sparse::CscMatrix;

/// ---------------------------- Number of genes per cell ----------------------------

pub fn chunked_num_genes<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Chunked num_genes: size {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_num_genes_chunk(chunk, &mut n_genes)?;
    }
    Ok(n_genes)
}

fn process_num_genes_chunk(csc_chunk: DynCscMatrix, n_genes: &mut Vec<u32>) -> anyhow::Result<()> {
    let csc: CscMatrix<f64> = csc_chunk.try_into()?;
    for &row_idx in csc.row_indices() {
        n_genes[row_idx] += 1;
    }
    Ok(())
}

pub fn compute_num_genes(csc: &DynCscMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_genes");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let mut n_genes = vec![0; csc_matrix.nrows()];
    for &row_idx in csc_matrix.row_indices() {
        n_genes[row_idx] += 1;
    }
    log::debug!("Num_genes computed");
    Ok(n_genes)
}

/// ---------------------------- Number of cells per gene ----------------------------

pub fn compute_num_cells(csc: &DynCscMatrix) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_cells");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let ret = csc_matrix
        .col_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32)
        .collect();
    log::debug!("Num_cells computed");
    Ok(ret)
}

pub fn chunked_num_cells<B: Backend>(x: &ArrayElem<B>, n_cols: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    log::debug!("Chunked num_cells: size {}", chunk_size);
    let mut n_cells = vec![0; n_cols];
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_num_cells_chunk(chunk, &mut n_cells)?;
    }
    Ok(n_cells)
}

fn process_num_cells_chunk(csc_chunk: DynCscMatrix, n_cells: &mut Vec<u32>) -> anyhow::Result<()> {
    let csc_matrix: CscMatrix<f64> = csc_chunk.try_into()?;
    let chunk_n_cells = csc_matrix.col_offsets().windows(2).map(|w| (w[1] - w[0]) as u32);
    for (i, count) in chunk_n_cells.enumerate() {
        n_cells[i] = count;
    }
    Ok(())
}

/// ---------------------------- Number of cells per gene AND number of genes per cell ----------------------------

pub fn chunked_num_combined<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Chunked num_combined: size {}", chunk_size);
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_num_combined_chunk(&chunk, &mut n_genes, &mut n_cells)?;
    }

    Ok((n_genes, n_cells))
}

fn process_num_combined_chunk(csc_chunk: &DynCscMatrix, n_genes: &mut Vec<u32>, n_cells: &mut Vec<u32>) -> anyhow::Result<()> {
    let csc: CscMatrix<f64> = csc_chunk.clone().try_into()?;
    
    for (col, col_vec) in csc.col_iter().enumerate() {
        n_cells[col] = col_vec.nnz() as u32;
        for &row in col_vec.row_indices() {
            n_genes[row] += 1;
        }
    }
    Ok(())
}

pub fn compute_num_combined(
    csc: &DynCscMatrix,
    n_rows: usize,
    n_cols: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Computing num_combined");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let mut n_genes = vec![0; n_rows];
    let mut n_cells = vec![0; n_cols];

    for (col, col_vec) in csc_matrix.col_iter().enumerate() {
        n_cells[col] = col_vec.nnz() as u32;
        for &row in col_vec.row_indices() {
            n_genes[row] += 1;
        }
    }

    log::debug!("Num_combined computed");
    Ok((n_genes, n_cells))
}

// ---------------------------- Sum of expression values per cell ----------------------------

pub fn chunked_sum_cell_expr<T: ArrayElemOp>(
    x: &T,
    n_rows: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<f64>> {
    log::debug!("Chunked sum_cell_expr: size {}", chunk_size);
    let mut gene_sums = vec![0.0; n_rows];
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_sum_cell_expr_chunk(chunk, &mut gene_sums)?;
    }
    Ok(gene_sums)
}

fn process_sum_cell_expr_chunk(csc_chunk: DynCscMatrix, gene_sums: &mut Vec<f64>) -> anyhow::Result<()> {
    let csc: CscMatrix<f64> = csc_chunk.try_into()?;
    for (row_idx, &value) in csc.row_indices().iter().zip(csc.values()) {
        gene_sums[*row_idx] += value;
    }
    Ok(())
}

pub fn compute_sum_cell_expr(csc: &DynCscMatrix) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_cell_expr");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let mut gene_sums = vec![0.0; csc_matrix.nrows()];
    for (row_idx, &value) in csc_matrix.row_indices().iter().zip(csc_matrix.values()) {
        gene_sums[*row_idx] += value;
    }
    log::debug!("Sum_cell_expr computed");
    Ok(gene_sums)
}

// ---------------------------- Sum of expression values per gene ----------------------------

pub fn chunked_sum_gene_expr<T: ArrayElemOp>(
    x: &T,
    n_cols: usize,
    chunk_size: usize,
) -> anyhow::Result<Vec<f64>> {
    log::debug!("Chunked sum_gene_expr: size {}", chunk_size);
    let mut cell_sums = vec![0.0; n_cols];
    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_sum_gene_expr_chunk(chunk, &mut cell_sums)?;
    }
    Ok(cell_sums)
}

fn process_sum_gene_expr_chunk(csc_chunk: DynCscMatrix, cell_sums: &mut Vec<f64>) -> anyhow::Result<()> {
    let csc_matrix: CscMatrix<f64> = csc_chunk.try_into()?;
    for (col, col_vec) in csc_matrix.col_iter().enumerate() {
        cell_sums[col] = col_vec.values().iter().sum();
    }
    Ok(())
}

pub fn compute_sum_gene_expr(csc: &DynCscMatrix) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_gene_expr");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let cell_sums: Vec<f64> = csc_matrix
        .col_iter()
        .map(|col| col.values().iter().sum())
        .collect();
    log::debug!("Sum_gene_expr computed");
    Ok(cell_sums)
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

    for (chunk, start, end) in x.iter::<DynCscMatrix>(chunk_size) {
        log::debug!("Processing chunk: {}-{}", start, end);
        process_sum_combined_chunk(chunk, &mut cell_sums, &mut gene_sums)?;
    }

    Ok((cell_sums, gene_sums))
}

fn process_sum_combined_chunk(
    csc_chunk: DynCscMatrix,
    cell_sums: &mut Vec<f64>,
    gene_sums: &mut Vec<f64>,
) -> anyhow::Result<()> {
    let csc: CscMatrix<f64> = csc_chunk.try_into()?;
    for (col, col_vec) in csc.col_iter().enumerate() {
        for (&row, &value) in col_vec.row_indices().iter().zip(col_vec.values()) {
            cell_sums[row] += value;
            gene_sums[col] += value;
        }
    }
    Ok(())
}

pub fn compute_sum_combined(
    csc: &DynCscMatrix,
    n_rows: usize,
    n_cols: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    log::debug!("Computing sum_combined");
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into()?;
    let mut cell_sums = vec![0.0; n_rows];
    let mut gene_sums = vec![0.0; n_cols];

    for (col, col_vec) in csc_matrix.col_iter().enumerate() {
        for (&row, &value) in col_vec.row_indices().iter().zip(col_vec.values()) {
            cell_sums[row] += value;
            gene_sums[col] += value;
        }
    }

    log::debug!("Sum_combined computed");
    Ok((cell_sums, gene_sums))
}

// ---------------------------- Num and Sum of expression values per gene ----------------------------