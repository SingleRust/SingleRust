use anndata::{AnnData, Backend};

use super::ComputationMode;
mod utils;



pub fn compute_n_genes<B: Backend>(
    anndata: &AnnData<B>,
    computation: ComputationMode,
) -> anyhow::Result<Vec<u32>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_num_genes_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_num_genes(anndata),
    }
}

pub fn compute_n_cells<B: Backend>(
    anndata: &AnnData<B>,
    computation: ComputationMode,
) -> anyhow::Result<Vec<u32>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_num_cells_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_num_cells(anndata),
    }
}

pub fn compute_n_combined<B: Backend>(
    anndata: &AnnData<B>,
    computation: ComputationMode,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_num_combined_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_num_combined(anndata),
    }
}

pub fn compute_sum_genes<B: Backend>(
    anndata: &AnnData<B>,
    computation: ComputationMode,
) -> anyhow::Result<Vec<f64>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_sum_gene_expr_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_sum_gene_expr(anndata),
    }
}

pub fn compute_sum_cells<B: Backend>(
    anndata: &AnnData<B>,
    computation: ComputationMode,
) -> anyhow::Result<Vec<f64>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_sum_cell_expr_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_sum_cell_expr(anndata),
    }
}

pub fn compute_sum_combined<B: Backend>(
    anndata: &AnnData<B>,
    computation: ComputationMode,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_sum_combined_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_sum_combined(anndata),
    }
}
