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
use log::info;

pub fn compute_n_genes_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
    n_threads: usize
) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
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
    
    let thread_results: Vec<Vec<u32>> = chunks.into_par_iter()
        .enumerate()
        .map(|(chunk_index, (chunk, start, _))| {
            let thread_id = rayon::current_thread_index().unwrap();
            info!("Thread {} processing chunk {}", thread_id, chunk_index);

            let csr: CsrMatrix<f64> = chunk.try_into().unwrap();
            let mut local_counts = vec![0; n_rows];
            
            csr.row_offsets().windows(2).enumerate().for_each(|(i, w)| {
                let count = (w[1] - w[0]) as u32;
                local_counts[start + i] = count;
            });

            info!("Thread {} finished chunk {}", thread_id, chunk_index);
            local_counts
        })
        .collect();

    // Merge results from all threads
    let mut n_genes = vec![0; n_rows];
    for local_counts in thread_results {
        for (i, count) in local_counts.into_iter().enumerate() {
            n_genes[i] = n_genes[i].max(count);  // Use max for CSR as we're overwriting, not adding
        }
    }

    Ok(n_genes)
}

fn compute_n_genes_csc_chunked<T: ArrayElemOp>(x: &T, n_rows: usize, chunk_size: usize) -> anyhow::Result<Vec<u32>> {
    let chunks: Vec<_> = x.iter::<DynCscMatrix>(chunk_size).collect();
    
    let thread_results: Vec<Vec<u32>> = chunks.into_par_iter()
        .enumerate()
        .map(|(chunk_index, (chunk, _, _))| {
            let thread_id = rayon::current_thread_index().unwrap();
            info!("Thread {} processing chunk {}", thread_id, chunk_index);

            let csc: CscMatrix<f64> = chunk.try_into().unwrap();
            let mut local_counts = vec![0; n_rows];
            
            for &row_idx in csc.row_indices() {
                local_counts[row_idx] += 1;
            }

            info!("Thread {} finished chunk {}", thread_id, chunk_index);
            local_counts
        })
        .collect();

    // Merge results from all threads
    let mut n_genes = vec![0; n_rows];
    for local_counts in thread_results {
        for (i, count) in local_counts.into_iter().enumerate() {
            n_genes[i] += count;  // Sum for CSC as we're accumulating counts
        }
    }

    Ok(n_genes)
}
