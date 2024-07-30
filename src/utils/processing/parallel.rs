use anndata::data::DynCscMatrix;
use anndata::data::DynCsrMatrix;
use anndata::AnnData;
use anndata::AnnDataOp;
use anndata::ArrayData;
use anndata::ArrayElemOp;
use anndata::Backend;
use nalgebra_sparse::CscMatrix;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use std::sync::Mutex;

pub fn compute_n_genes_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
    n_threads: usize
) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_rows = shape[0];
    
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))?;

    pool.install(|| {
        match x.get::<ArrayData>()? {
            Some(ArrayData::CscMatrix(_)) => compute_n_genes_csc_chunked(&x, n_rows, chunk_size),
            Some(ArrayData::CsrMatrix(_)) => compute_n_genes_csr_chunked(&x, n_rows, chunk_size),
            _ => Err(anyhow::anyhow!("Unsupported array type for X")),
        }
    })
}

fn compute_n_genes_csr_chunked<T: ArrayElemOp>(x: &T, n_rows: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    let chunks: Vec<_> = x.iter::<DynCsrMatrix>(chunk_size).collect();
    let n_genes = Arc::new(Mutex::new(vec![0; n_rows]));

    chunks.into_par_iter().try_for_each(|(chunk, start, _)| {
        let csr: CsrMatrix<f64> = chunk.try_into()?;
        let chunk_n_genes: Vec<u32> = csr.row_offsets()
            .windows(2)
            .map(|w| (w[1] - w[0]) as u32)
            .collect();
        
        let mut n_genes = n_genes.lock().map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        for (i, count) in chunk_n_genes.into_iter().enumerate() {
            n_genes[start + i] = count;
        }
        Ok::<_, anyhow::Error>(())
    })?;

    Arc::try_unwrap(n_genes)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap Arc"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("Failed to unwrap Mutex"))
}

fn compute_n_genes_csc_chunked<T: ArrayElemOp>(x: &T, n_rows: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    let chunks: Vec<_> = x.iter::<DynCscMatrix>(chunk_size).collect();
    let n_genes = Arc::new(Mutex::new(vec![0; n_rows]));

    chunks.into_par_iter().try_for_each(|(chunk, _, _)| {
        let csc: CscMatrix<f64> = chunk.try_into()?;
        let mut local_n_genes = vec![0; n_rows];
        for &row_idx in csc.row_indices() {
            local_n_genes[row_idx] += 1;
        }
        
        let mut n_genes = n_genes.lock().map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        for (i, count) in local_n_genes.into_iter().enumerate() {
            n_genes[i] += count;
        }
        Ok::<_, anyhow::Error>(())
    })?;

    Arc::try_unwrap(n_genes)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap Arc"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("Failed to unwrap Mutex"))
}
