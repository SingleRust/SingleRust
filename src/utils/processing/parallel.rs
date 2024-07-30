use anndata::{
    data::{DynCscMatrix, DynCsrMatrix},
    AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend,
};
use anyhow::Ok;
use nalgebra_sparse::{csr::CsrMatrix, CscMatrix};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::Mutex;
use std::sync::atomic::AtomicU32;
use itertools::Itertools;
use rayon::ThreadPoolBuilder;

#[cfg(feature = "parallel")]
fn compute_n_cells_csc_chunked_parallel<T: ArrayElemOp>(
    x: &T, 
    n_cols: usize,
    thread_count: usize
) -> anyhow::Result<Vec<u32>> {
    use std::sync::atomic::Ordering;

    

    let n_cells: Vec<AtomicU32> = (0..n_cols).map(|_| AtomicU32::new(0)).collect();

    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))?;

    pool.install(|| {
        x.iter::<DynCscMatrix>(1000).try_for_each(|(chunk, start, _)| {
            let csc: CscMatrix<f64> = chunk.try_into()?;
            let chunk_n_cells: Vec<u32> = csc.col_offsets()
                .windows(2)
                .map(|w| (w[1] - w[0]) as u32)
                .collect();

            chunk_n_cells.par_iter().enumerate().for_each(|(i, &count)| {
                n_cells[start + i].store(count, Ordering::Relaxed);
            });

            Ok(())
        })
    })?;

    Ok(n_cells.into_iter().map(|atomic| atomic.into_inner()).collect())
}

#[cfg(feature = "parallel")]
fn compute_n_cells_csr_chunked_parallel<T: ArrayElemOp>(
    x: &T, 
    n_cols: usize,
    thread_count: usize
) -> anyhow::Result<Vec<u32>> {
    use std::sync::atomic::Ordering;

    let n_cells: Vec<AtomicU32> = (0..n_cols).map(|_| AtomicU32::new(0)).collect();

    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create thread pool: {}", e))?;

    pool.install(|| {
        x.iter::<DynCsrMatrix>(1000).try_for_each(|(chunk, _, _)| {
            let csr: CsrMatrix<f64> = chunk.try_into()?;
            
            csr.col_indices().par_iter().for_each(|&col_idx| {
                n_cells[col_idx].fetch_add(1, Ordering::Relaxed);
            });
            
            Ok(())
        })
    })?;

    Ok(n_cells.into_iter().map(|atomic| atomic.into_inner()).collect())
}