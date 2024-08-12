use anndata::{
    data::{DynCscMatrix, DynCsrMatrix},
    AnnData, AnnDataOp, ArrayData, ArrayElemOp, ArrayOp, Backend,
};
use anyhow::anyhow;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};

use crate::{statistics, utils::ComputationMode};

mod helpers;

pub fn normalize_per_cell<B: Backend>(
    anndata: &AnnData<B>,
    target_sum: f64,
    mode: ComputationMode,
) -> anyhow::Result<()> {
    let cell_sums = statistics::compute_sum_cells(anndata, mode.clone())?;

    match mode {
        ComputationMode::Chunked(chunk_size) => {
            normalize_per_cell_chunked(anndata, target_sum, chunk_size, &cell_sums)
        }
        ComputationMode::Whole => normalize_per_cell_whole(anndata, target_sum, &cell_sums),
    }
}

fn normalize_per_cell_whole<B: Backend>(
    anndata: &AnnData<B>,
    target_sum: f64,
    cell_sums: &[f64],
) -> anyhow::Result<()> {
    let x = anndata
        .x()
        .get::<ArrayData>()?
        .ok_or_else(|| anyhow!("X is empty!"))?;

    let normalized_matrix = match x {
        ArrayData::CsrMatrix(matrix) => normalize_chunk_csr(matrix, target_sum, cell_sums)?,
        ArrayData::CscMatrix(matrix) => normalize_chunk_csc(matrix, target_sum, cell_sums)?,
        _ => return Err(anyhow!("Unsupported matrix type for normalization")),
    };

    anndata.set_x(normalized_matrix)
}

fn normalize_chunk_csr(
    matrix: DynCsrMatrix,
    target_sum: f64,
    cell_sums: &[f64],
) -> anyhow::Result<ArrayData> {
    let csr_matrix: CsrMatrix<f64> = matrix.try_into()?;
    let nrows = csr_matrix.nrows();
    let ncols = csr_matrix.ncols();

    // Create a CooMatrix to collect normalized data
    let mut coo_normalized = CooMatrix::<f64>::new(nrows, ncols);

    for (row_idx, (row, &sum)) in csr_matrix.row_iter().zip(cell_sums.iter()).enumerate() {
        for col_val_index in 0..row.nnz() {
            let col_ind = row.col_indices()[col_val_index];
            let val = row.values()[col_val_index];
            let normalized_val = val * target_sum / sum;
            coo_normalized.push(row_idx, col_ind, normalized_val);
        }
    }

    // Convert CooMatrix back to CsrMatrix
    let normalized_matrix: CsrMatrix<f64> = CsrMatrix::from(&coo_normalized);

    Ok(ArrayData::CsrMatrix(normalized_matrix.into()))
}

fn normalize_chunk_csc(
    matrix: DynCscMatrix,
    target_sum: f64,
    cell_sums: &[f64],
) -> anyhow::Result<ArrayData> {
    let csc_matrix: CscMatrix<f64> = matrix.try_into()?;
    let nrows = csc_matrix.nrows();
    let ncols = csc_matrix.ncols();

    // Create a CooMatrix to collect normalized data
    let mut coo_normalized = CooMatrix::new(nrows, ncols);

    for (col_idx, col) in csc_matrix.col_iter().enumerate() {
        for j in 0..col.nnz() {
            let row = col.row_indices()[j];
            let val = col.values()[j];
            let normalized_val = val * target_sum / cell_sums[row];
            coo_normalized.push(row, col_idx, normalized_val);
        }
    }

    // Convert CooMatrix back to CscMatrix
    let normalized_matrix: CscMatrix<f64> = CscMatrix::from(&coo_normalized);

    Ok(ArrayData::CscMatrix(normalized_matrix.into()))
}

fn normalize_per_cell_chunked<B: Backend>(
    anndata: &AnnData<B>,
    target_sum: f64,
    chunk_size: usize,
    cell_sums: &[f64],
) -> anyhow::Result<()> {
    let x = anndata.x();
    let mut normalized_data = Vec::new();
    let mut cell_index = 0;

    for (chunk, start, end) in x.iter::<ArrayData>(chunk_size) {
        let chunk_sums = &cell_sums[cell_index..cell_index + (end - start)];
        let normalized_chunk = match chunk {
            ArrayData::CsrMatrix(matrix) => normalize_chunk_csr(matrix, target_sum, chunk_sums),
            ArrayData::CscMatrix(matrix) => normalize_chunk_csc(matrix, target_sum, chunk_sums),
            _ => return Err(anyhow!("Unsupported matrix type for normalization")),
        }?;
        normalized_data.push(normalized_chunk);
        cell_index += end - start;
    }

    // Combine normalized chunks and update X
    let combined_matrix = ArrayData::vstack(normalized_data.into_iter())?;
    anndata.set_x(combined_matrix)
}

pub fn normalize_per_gene<B: Backend>(
    anndata: &AnnData<B>,
    target_sum: f64,
    mode: ComputationMode,
) -> anyhow::Result<()> {
    let gene_sums = statistics::compute_sum_genes(anndata, mode.clone())?;

    match mode {
        ComputationMode::Chunked(chunk_size) => {
            normalize_per_gene_chunked(anndata, target_sum, chunk_size, &gene_sums)
        }
        ComputationMode::Whole => normalize_per_gene_whole(anndata, target_sum, &gene_sums),
    }
}

fn normalize_per_gene_chunked<B: Backend>(
    anndata: &AnnData<B>,
    target_sum: f64,
    chunk_size: usize,
    gene_sums: &[f64],
) -> anyhow::Result<()> {
    let x = anndata.x();
    let n_obs = anndata.n_obs();

    let normalized_iter =
        x.iter::<ArrayData>(chunk_size)
            .enumerate()
            .flat_map(move |(chunk_index, chunk_result)| {
                let chunk = chunk_result.unwrap();
                let start = chunk_index * chunk_size;
                let end = (start + chunk.shape()[1]).min(gene_sums.len());
                let chunk_sums = &gene_sums[start..end];

                let normalized_chunk = match chunk {
                    ArrayData::CsrMatrix(matrix) => {
                        normalize_gene_chunk_csr(matrix, target_sum, chunk_sums)
                    }
                    ArrayData::CscMatrix(matrix) => {
                        normalize_gene_chunk_csc(matrix, target_sum, chunk_sums)
                    }
                    _ => panic!("Unsupported matrix type for normalization"),
                }
                .unwrap();

                normalized_chunk
            });

    let combined_matrix = ArrayData::vstack(normalized_data.into_iter())?;
    anndata.set_x(combined_matrix)
}

fn normalize_per_gene_whole<B: Backend>(
    anndata: &AnnData<B>,
    target_sum: f64,
    gene_sums: &[f64],
) -> anyhow::Result<()> {
    let x: ArrayData = anndata.x().get()?.ok_or_else(|| anyhow!("X is empty"))?;

    let normalized_matrix = match x {
        ArrayData::DynCsr(matrix) => normalize_gene_chunk_csr(matrix, target_sum, gene_sums)?,
        ArrayData::DynCsc(matrix) => normalize_gene_chunk_csc(matrix, target_sum, gene_sums)?,
        _ => return Err(anyhow!("Unsupported matrix type for normalization")),
    };

    anndata.set_x(normalized_matrix)
}

fn normalize_gene_chunk_csr(
    matrix: DynCsrMatrix,
    target_sum: f64,
    gene_sums: &[f64],
) -> anyhow::Result<ArrayData> {
    let csr_matrix: CsrMatrix<f64> = matrix.try_into()?;
    let nrows = csr_matrix.nrows();
    let ncols = csr_matrix.ncols();

    let mut coo_normalized = CooMatrix::new(nrows, ncols);

    for (row_idx, row) in csr_matrix.row_iter().enumerate() {
        for j in 0..row.nnz() {
            let col = row.col_indices()[j];
            let val = row.values()[j];
            let normalized_val = val * target_sum / gene_sums[col];
            coo_normalized.push(row_idx, col, normalized_val);
        }
    }

    let normalized_matrix: CsrMatrix<f64> = CsrMatrix::from(coo_normalized);

    Ok(ArrayData::CsrMatrix(normalized_matrix.into()))
}

fn normalize_gene_chunk_csc(
    matrix: DynCscMatrix,
    target_sum: f64,
    gene_sums: &[f64],
) -> anyhow::Result<ArrayData> {
    let csc_matrix: CscMatrix<f64> = matrix.try_into()?;
    let nrows = csc_matrix.nrows();
    let ncols = csc_matrix.ncols();

    let mut coo_normalized = CooMatrix::new(nrows, ncols);

    for (col_idx, col) in csc_matrix.col_iter().enumerate() {
        let gene_sum = gene_sums[col_idx];
        for j in 0..col.nnz() {
            let row = col.row_indices()[j];
            let val = col.values()[j];
            let normalized_val = val * target_sum / gene_sum;
            coo_normalized.push(row, col_idx, normalized_val);
        }
    }

    let normalized_matrix: CscMatrix<f64> = coo_normalized.into();

    Ok(ArrayData::DynCsc(normalized_matrix.into()))
}
