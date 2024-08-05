use anndata::{
    backend::DataType, AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend
};
mod csc;
mod csr;

pub fn compute_num_genes<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_genes");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    let n_genes = match x {
        ArrayData::CscMatrix(csc) => csc::compute_num_genes(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_num_genes(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    };
    n_genes
}

pub fn compute_num_genes_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_genes chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_rows = shape[0];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_num_genes(&x, n_rows, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_num_genes(&x, n_rows, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X"))
    }
}

pub fn compute_num_cells<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_cells");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_num_cells(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_num_cells(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_num_cells_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_cells chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_num_cells(&x, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_num_cells(&x, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_num_combined<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Computing num_combined");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    let shape = adata.x().shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_num_combined(&csc, n_rows, n_cols),
        ArrayData::CsrMatrix(csr) => csr::compute_num_combined(&csr, n_rows, n_cols),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_num_combined_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Computing num_combined chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_num_combined(&x, n_rows, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_num_combined(&x, n_rows, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

// ------------------------------------------- SUM CALC -------------------------------------------

pub fn compute_sum_gene_expr<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_gene_expr");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_sum_gene_expr(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_sum_gene_expr(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_sum_gene_expr_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_gene_expr chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_rows = shape[0];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_sum_gene_expr(&x, n_rows, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_sum_gene_expr(&x, n_rows, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_sum_cell_expr<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_cell_expr");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_sum_cell_expr(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_sum_cell_expr(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_sum_cell_expr_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<f64>> {
    log::debug!("Computing sum_cell_expr chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_sum_cell_expr(&x, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_sum_cell_expr(&x, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_sum_combined<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    log::debug!("Computing sum_combined");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    let shape = adata.x().shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_sum_combined(&csc, n_rows, n_cols),
        ArrayData::CsrMatrix(csr) => csr::compute_sum_combined(&csr, n_rows, n_cols),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_sum_combined_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    log::debug!("Computing sum_combined chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_sum_combined(&x, n_rows, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_sum_combined(&x, n_rows, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}