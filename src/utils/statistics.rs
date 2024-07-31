use anndata::{
    data::{DynCscMatrix, DynCsrMatrix}, AnnData, AnnDataOp, ArrayData, ArrayElem, ArrayElemOp, Backend
};
use anyhow::Ok;
use nalgebra_sparse::{csr::CsrMatrix, CscMatrix};
mod utils;

pub enum ComputationMode {
    Chunked(usize),
    Whole
}

pub fn compute_n_genes<B: Backend>(anndata: &AnnData<B>, computation: ComputationMode) -> annhow::Result<Vec<u32>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_n_genes_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_n_genes(anndata),
    }
}

pub fn compute_n_cells<B: Backend>(anndata: &AnnData<B>, computation: ComputationMode) -> anyhow::Result<Vec<u32>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_n_cells_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_n_cells(anndata),
    }
}

pub fn compute_n_combined<B: Backend>(anndata: &AnnData<B>, computation: ComputationMode) -> anyhow::Result<Vec<u32>> {
    match computation {
        ComputationMode::Chunked(size) => utils::compute_n_combined_chunked(anndata, size),
        ComputationMode::Whole => utils::compute_n_combined(anndata),
    }
}

pub fn compute_sum_genes<B: Backend>(anndata: &AnnData<B>, computation: ComputationMode) -> anyhow::Result<Vec<u32>> {

}

pub fn compute_sum_cells<B: Backend>(anndata: &AnnData<B>, computation: ComputationMode) -> anyhow::Result<Vec<u32>> {
    
}

pub fn compute_sum_combined<B: Backend>(anndata: &AnnData<B>, computation: ComputationMode) -> anyhow::Result<Vec<u32>> {

}
