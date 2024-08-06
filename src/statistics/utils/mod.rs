use anndata::{backend::DataType, data::{array, Shape}, AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
mod csc;
mod csr;

/// Computes the number of genes expressed in each cell.
///
/// This function calculates the number of non-zero entries in each row of the gene expression matrix.
///
/// # Arguments
///
/// * `adata` - An AnnData object containing the gene expression data.
///
/// # Returns
///
/// A vector of u32 values, where each value represents the number of expressed genes for a cell.
///
/// # Errors
///
/// Returns an error if the X matrix is not found or if the array type is unsupported.
pub fn compute_num_genes<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_genes");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_num_genes(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_num_genes(&csr),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

/// Computes the number of genes expressed in each cell using a chunked approach.
///
/// This function calculates the number of non-zero entries in each row of the gene expression matrix,
/// processing the data in chunks to reduce memory usage.
///
/// # Arguments
///
/// * `adata` - An AnnData object containing the gene expression data.
/// * `chunk_size` - The number of cells to process in each chunk.
///
/// # Returns
///
/// A vector of u32 values, where each value represents the number of expressed genes for a cell.
///
/// # Errors
///
/// Returns an error if the X matrix shape is not found or if the array type is unsupported.
pub fn compute_num_genes_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_genes chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_rows = shape[0];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_num_genes(&x, n_rows, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_num_genes(&x, n_rows, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

/// Computes the number of cells expressing each gene.
///
/// This function calculates the number of non-zero entries in each column of the gene expression matrix.
///
/// # Arguments
///
/// * `adata` - An AnnData object containing the gene expression data.
///
/// # Returns
///
/// A vector of u32 values, where each value represents the number of cells expressing a particular gene.
///
/// # Errors
///
/// Returns an error if the X matrix is not found or if the array type is unsupported.
pub fn compute_num_cells<B: Backend>(adata: &AnnData<B>) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_cells");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_num_cells(&csc),
        ArrayData::CsrMatrix(csr) => csr::compute_num_cells(&csr),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

/// Computes the number of cells expressing each gene using a chunked approach.
///
/// This function calculates the number of non-zero entries in each column of the gene expression matrix,
/// processing the data in chunks to reduce memory usage.
///
/// # Arguments
///
/// * `adata` - An AnnData object containing the gene expression data.
/// * `chunk_size` - The number of genes to process in each chunk.
///
/// # Returns
///
/// A vector of u32 values, where each value represents the number of cells expressing a particular gene.
///
/// # Errors
///
/// Returns an error if the X matrix shape is not found or if the array type is unsupported.
pub fn compute_num_cells_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<Vec<u32>> {
    log::debug!("Computing num_cells chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_num_cells(&x, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_num_cells(&x, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

/// Computes both the number of genes expressed in each cell and the number of cells expressing each gene.
///
/// This function calculates the number of non-zero entries in each row and each column of the gene expression matrix.
///
/// # Arguments
///
/// * `adata` - An AnnData object containing the gene expression data.
///
/// # Returns
///
/// A tuple of two vectors:
/// * The first vector contains u32 values representing the number of expressed genes for each cell.
/// * The second vector contains u32 values representing the number of cells expressing each gene.
///
/// # Errors
///
/// Returns an error if the X matrix is not found, if the matrix shape is not found, or if the array type is unsupported.
pub fn compute_num_combined<B: Backend>(
    adata: &AnnData<B>,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Computing num_combined");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    let shape = adata
        .x()
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    match x {
        ArrayData::CscMatrix(csc) => csc::compute_num_combined(&csc, n_rows, n_cols),
        ArrayData::CsrMatrix(csr) => csr::compute_num_combined(&csr, n_rows, n_cols),
        _ => return Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

/// Computes both the number of genes expressed in each cell and the number of cells expressing each gene using a chunked approach.
///
/// This function calculates the number of non-zero entries in each row and each column of the gene expression matrix,
/// processing the data in chunks to reduce memory usage.
///
/// # Arguments
///
/// * `adata` - An AnnData object containing the gene expression data.
/// * `chunk_size` - The number of cells/genes to process in each chunk.
///
/// # Returns
///
/// A tuple of two vectors:
/// * The first vector contains u32 values representing the number of expressed genes for each cell.
/// * The second vector contains u32 values representing the number of cells expressing each gene.
///
/// # Errors
///
/// Returns an error if the X matrix shape is not found or if the array type is unsupported.
pub fn compute_num_combined_chunked<B: Backend>(
    adata: &AnnData<B>,
    chunk_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    log::debug!("Computing num_combined chunked: size {}", chunk_size);
    let x = adata.x();
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_num_combined(&x, n_rows, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_num_combined(&x, n_rows, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

// ------------------------------------------- SUM CALC -------------------------------------------

/// Computes the sum of gene expressions from the given `AnnData` object.
///
/// This function retrieves the `X` matrix from the `AnnData` object and computes the sum of gene expressions
/// based on the type of the matrix (CSC or CSR).
///
/// # Type Parameters
/// - `B`: A type that implements the `Backend` trait.
///
/// # Arguments
/// - `adata`: A reference to an `AnnData` object containing the gene expression data.
///
/// # Returns
/// - `anyhow::Result<Vec<f64>>`: A result containing a vector of gene expression sums if successful, or an error if the matrix type is unsupported or if there is an issue retrieving the matrix.
///
/// # Errors
/// - Returns an error if the `X` matrix is not found in the `AnnData` object.
/// - Returns an error if the `X` matrix type is unsupported.
///
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
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
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
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let n_cols = shape[1];
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_sum_cell_expr(&x, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_sum_cell_expr(&x, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

pub fn compute_sum_combined<B: Backend>(
    adata: &AnnData<B>,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    log::debug!("Computing sum_combined");
    let x = adata.x().get::<ArrayData>()?.expect("X matrix not found");
    let shape = adata
        .x()
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
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
    let shape = x
        .shape()
        .ok_or_else(|| anyhow::anyhow!("X matrix shape not found"))?;
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let array_type = x.inner().dtype();
    match array_type {
        DataType::CscMatrix(_) => csc::chunked_sum_combined(&x, n_rows, n_cols, chunk_size),
        DataType::CsrMatrix(_) => csr::chunked_sum_combined(&x, n_rows, n_cols, chunk_size),
        _ => Err(anyhow::anyhow!("Unsupported array type for X")),
    }
}

// pub fn compute_num_sum_cells_chunked<B: Backend>(adata: &AnnData<B>, chunk_size: usize) -> anyhow::Result<(Vec<u32>, Vec<f64>)> {
//     let x = adata.x();
//     let shape = x.shape().ok_or_else(|| anyhow::anyhow!("Failed to get shape of matrix X!"))?;
//     let n_cols = shape[0];
//     let array_type = x.inner().dtype();
//     match array_type {
//         DataType::CscMatrix(_) => csc::chunked_sum_combined(&x, n_rows, n_cols, chunk_size),
//         DataType::CsrMatrix(_) => csr::chunked_sum_combined(&x, n_rows, n_cols, chunk_size),
//         _ => Err(anyhow::anyhow!("Unsupported array type for X")),
//     }
// }
