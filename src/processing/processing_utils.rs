use anndata::{
    data::{DynCscMatrix, DynCsrMatrix},
    AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend,
};
use nalgebra_sparse::{csr::CsrMatrix, CscMatrix};

pub fn compute_n_genes_chunked<B: Backend>(adata: &mut AnnData<B>) -> anyhow::Result<Vec<u32>> {
    let x = adata.x();
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_rows = shape[0];

    match x.get::<ArrayData>()? {
        Some(ArrayData::CscMatrix(_)) => compute_n_genes_csc_chunked(&x, n_rows),
        Some(ArrayData::CsrMatrix(_)) => compute_n_genes_csr_chunked(&x, n_rows),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_n_genes<B: Backend>(adata: &mut AnnData<B>) -> anyhow::Result<Vec<u32>> {
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");

    let n_genes = match x {
        ArrayData::CscMatrix(csc) => compute_n_genes_csc(&csc),
        ArrayData::CsrMatrix(csr) => compute_n_genes_csr(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    };

    Ok(n_genes)
}

fn compute_n_genes_csr(csr: &DynCsrMatrix) -> Vec<u32> {
    let csr_matrix: CsrMatrix<f64> = csr.clone().try_into().expect("Could not convert data.");
    csr_matrix
        .row_offsets()
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32)
        .collect()
}

fn compute_n_genes_csr_chunked<T: ArrayElemOp>(x: &T, n_rows: usize) -> anyhow::Result<Vec<u32>> {
    let mut n_genes = vec![0; n_rows];

    for (chunk, start, _) in x.iter::<DynCsrMatrix>(1000) {
        let csr: CsrMatrix<f64> = chunk.try_into()?;
        let chunk_n_genes = csr.row_offsets().windows(2).map(|w| (w[1] - w[0]) as u32);

        for (i, count) in chunk_n_genes.enumerate() {
            n_genes[start + i] = count;
        }
    }
    Ok(n_genes)
}

fn compute_n_genes_csc_chunked<T: ArrayElemOp>(x: &T, n_rows: usize) -> anyhow::Result<Vec<u32>> {
    let mut n_genes = vec![0; n_rows];

    for (chunk, _, _) in x.iter::<DynCscMatrix>(1000) {
        let csc: CscMatrix<f64> = chunk.try_into()?;
        for &row_idx in csc.row_indices() {
            n_genes[row_idx] += 1;
        }
    }
    Ok(n_genes)
}

fn compute_n_genes_csc(csc: &DynCscMatrix) -> Vec<u32> {
    let csc_matrix: CscMatrix<f64> = csc.clone().try_into().expect("Could not convert data.");
    let mut n_genes = vec![0; csc_matrix.nrows()];
    for &row_idx in csc_matrix.row_indices() {
        n_genes[row_idx] += 1;
    }
    n_genes
}
