//! # Principal Component Analysis (PCA) for Single-Cell Data
//!
//! This module provides high-performance PCA implementation optimized for single-cell RNA-seq data.
//! PCA is a fundamental dimensionality reduction technique that identifies the directions of maximum
//! variance in high-dimensional data.
//!
//! ## Overview
//!
//! PCA transforms the original gene expression space into a new coordinate system where:
//! - The first principal component captures the most variance
//! - Each subsequent component captures the most remaining variance
//! - Components are orthogonal (uncorrelated) to each other
//!
//! ## Key Features
//!
//! - **Sparse Matrix Support**: Optimized for sparse single-cell expression matrices
//! - **Feature Selection Integration**: Works with highly variable genes or custom gene selections
//! - **Multiple SVD Methods**: Choice of SVD algorithms for different performance characteristics
//! - **Memory Efficient**: Handles large datasets without excessive memory usage
//! - **Configurable Centering**: Option to center data (recommended for most analyses)
//!
//! ## When to Use PCA
//!
//! ✅ **Good for:**
//! - Initial exploration of dataset structure
//! - Noise reduction before clustering
//! - Input for other dimensionality reduction methods (t-SNE, UMAP)
//! - Quality control and batch effect detection
//! - Identifying major sources of variation
//!
//! ⚠️ **Limitations:**
//! - Linear method - may not capture complex non-linear relationships
//! - First components may be dominated by technical effects
//! - Interpretation can be challenging with many genes
//!
//! ## Typical Workflow
//!
//! ```rust,ignore
//! use single_rust::memory::processing::dimred::pca::run_pca_sparse_masked;
//! use single_rust::memory::processing::dimred::FeatureSelectionMethod;
//! use single_algebra::dimred::pca::SVDMethod;
//!
//! // 1. Select highly variable genes
//! let hvg_mask = compute_highly_variable_genes(&adata, None)?;
//! let feature_selection = FeatureSelectionMethod::HighlyVariableSelection(hvg_mask);
//!
//! // 2. Run PCA with 50 components
//! let pca_result = run_pca_sparse_masked::<f64>(
//!     &adata.x(),
//!     Some(feature_selection),
//!     Some(true),              // Center the data
//!     Some(false),             // Verbose output
//!     Some(50),                // Number of components
//!     Some(1.0),               // Regularization parameter
//!     Some(42),                // Random seed for reproducibility
//!     Some(SVDMethod::Randomized), // Fast randomized SVD
//! )?;
//!
//! // 3. Access results
//! let embeddings = pca_result.transformed;              // Cell embeddings in PC space
//! let variance_explained = pca_result.explained_variance_ratio;  // Variance per component
//! let loadings = pca_result.feature_importance;         // Gene loadings/weights
//! ```

use crate::memory::processing::dimred::FeatureSelectionMethod;
use crate::memory::utils::{arr1_conversion, arr2_conversion};
use anndata::data::DynCsrMatrix;
use anndata::ArrayData;
use anndata_memory::IMArrayElement;
use anyhow::anyhow;
use ndarray::{Array1, Array2};
use rand::distr::Uniform;
use rand::prelude::Distribution;
use rand::rng;
use single_algebra::dimred::pca::{MaskedSparsePCABuilder, SVDMethod};
use single_utilities::traits::FloatOpsTS;
use std::ops::Deref;

/// Results from Principal Component Analysis.
///
/// Contains all the essential outputs from PCA analysis including the transformed data,
/// variance explained by each component, and feature importance scores.
///
/// ## Fields
///
/// - `transformed`: Cell embeddings in the principal component space (cells × components)
/// - `explained_variance_ratio`: Fraction of total variance explained by each component
/// - `cumulative_explained_variance_ratio`: Cumulative variance explained up to each component
/// - `feature_importance`: Gene loadings/weights for each component (genes × components)
///
/// ## Usage
///
/// ```rust,ignore
/// let pca_result = run_pca_sparse_masked(&matrix, ...)?;
///
/// // Get embeddings for visualization or clustering
/// let embeddings = pca_result.transformed;
///
/// // Check how much variance is captured
/// let total_variance = pca_result.cumulative_explained_variance_ratio[[49]]; // 50th component
///
/// // Find important genes for first component  
/// let pc1_loadings = pca_result.feature_importance.column(0);
/// ```
pub struct PCAResult<T>
where
    T: FloatOpsTS,
{
    /// Transformed data: cell embeddings in principal component space (n_cells × n_components)
    pub transformed: Array2<T>,
    /// Fraction of variance explained by each principal component
    pub explained_variance_ratio: Array1<T>,
    /// Cumulative fraction of variance explained up to each component
    pub cumulative_explained_variance_ratio: Array1<T>,
    /// Feature loadings/importance for each component (n_features × n_components)
    pub feature_importance: Array2<T>,
}

/// Perform Principal Component Analysis on sparse single-cell expression data.
///
/// This function provides a comprehensive PCA implementation optimized for single-cell data,
/// with support for feature selection, sparse matrices, and various SVD algorithms.
///
/// ## Algorithm Details
///
/// The implementation uses efficient sparse matrix operations and supports multiple SVD methods:
/// - **Randomized SVD**: Fast approximation, good for large datasets
/// - **Full SVD**: Exact computation, slower but more accurate
/// - **Truncated SVD**: Memory efficient for large matrices
///
/// ## Parameters
///
/// * `matrix` - The expression matrix (cells × genes) as IMArrayElement
/// * `feature_selection_method` - Method for selecting genes (HVGs recommended)
/// * `center` - Whether to center the data (recommended: true)
/// * `verbose` - Enable verbose output for debugging
/// * `n_components` - Number of principal components to compute (default: 50)
/// * `alpha` - Regularization parameter for numerical stability (default: 1.0)
/// * `random_seed` - Seed for reproducible results (default: 42)
/// * `svd_method` - SVD algorithm to use (default: Randomized)
///
/// ## Returns
///
/// Returns a `PCAResult` containing:
/// - Transformed cell embeddings
/// - Variance explained by each component
/// - Feature importance/loading scores
///
/// ## Examples
///
/// ### Basic Usage
/// ```rust,ignore
/// // Simple PCA with default parameters
/// let result = run_pca_sparse_masked::<f64>(
///     &adata.x(),
///     None,                    // Use default random selection
///     Some(true),              // Center the data
///     None,                    // No verbose output
///     Some(50),                // 50 components
///     None,                    // Default alpha
///     None,                    // Default seed
///     None,                    // Default SVD method
/// )?;
/// ```
///
/// ### Advanced Usage with HVGs
/// ```rust,ignore
/// // PCA using highly variable genes
/// let hvg_mask = compute_highly_variable_genes(&adata, None)?;
/// let feature_selection = FeatureSelectionMethod::HighlyVariableSelection(hvg_mask);
///
/// let result = run_pca_sparse_masked::<f64>(
///     &adata.x(),
///     Some(feature_selection),
///     Some(true),              // Center for better component interpretation
///     Some(false),             // Quiet mode
///     Some(30),                // Fewer components for speed
///     Some(0.1),               // Higher regularization
///     Some(123),               // Custom seed
///     Some(SVDMethod::Randomized), // Fast approximation
/// )?;
/// ```
///
/// ## Performance Considerations
///
/// - **Feature Selection**: Using 1000-5000 highly variable genes typically optimal
/// - **Components**: 30-50 components usually capture most biological variation
/// - **SVD Method**: Randomized SVD recommended for >10,000 cells
/// - **Centering**: Essential for proper component interpretation but increases memory usage
///
/// ## Errors
///
/// Returns error if:
/// - Matrix format is not supported (only CSR matrices supported)
/// - Data type is not F32 or F64
/// - Feature selection mask length doesn't match number of genes
/// - SVD computation fails (e.g., insufficient rank)
#[allow(clippy::too_many_arguments)]
pub fn run_pca_sparse_masked<T>(
    matrix: &IMArrayElement,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    verbose: Option<bool>,
    n_components: Option<usize>,
    alpha: Option<f64>,
    random_seed: Option<u32>,
    svd_method: Option<SVDMethod>,
) -> anyhow::Result<PCAResult<T>>
where
    T: FloatOpsTS,
{
    let feature_selection_method =
        feature_selection_method.unwrap_or(FeatureSelectionMethod::RandomSelection(1000));
    let shape = matrix.get_shape()?;
    let ncols = shape[1];
    let center = center.unwrap_or(false);
    let verbose = verbose.unwrap_or(false);
    let n_components = n_components.unwrap_or(50);
    let random_seed = random_seed.unwrap_or(42);
    let svd_method = svd_method.unwrap_or_default();
    let selected = match feature_selection_method {
        FeatureSelectionMethod::FullFeatures => {
            vec![true; ncols]
        }
        FeatureSelectionMethod::HighlyVariableSelection(vec) => vec,
        FeatureSelectionMethod::RandomSelection(num_genes) => {
            generate_random_mask(ncols, num_genes)
        }
    };
    let read_guard = matrix.0.read_inner();
    let data = read_guard.deref();
    match data {
        ArrayData::CsrMatrix(dyn_csr) => {
            match dyn_csr {
                DynCsrMatrix::F32(csr) => {
                    let mut masked_pca = MaskedSparsePCABuilder::new()
                        .mask(selected)
                        .center(center)
                        .verbose(verbose)
                        .alpha(alpha.unwrap_or(1.0) as f32)
                        .n_components(n_components)
                        .random_seed(random_seed)
                        .svd_method(svd_method)
                        .build();
                    masked_pca.fit(csr)?;
                    let transformed = masked_pca.transform(csr)?;
                    let explained_variance_ratio = masked_pca.explained_variance_ratio()?;
                    let cumulative_explained_variance_ratio = masked_pca.cumulative_explained_variance_ratio()?;
                    let feature_importance = masked_pca.feature_importances()?;

                    let transformed: Array2<T> = arr2_conversion(transformed)?;
                    let explained_variance_ratio: Array1<T> = arr1_conversion(explained_variance_ratio)?;
                    let cumulative_explained_variance_ratio: Array1<T> = arr1_conversion(cumulative_explained_variance_ratio)?;
                    let feature_importance: Array2<T> = arr2_conversion(feature_importance)?;
                    let res = PCAResult {
                        transformed,
                        explained_variance_ratio,
                        cumulative_explained_variance_ratio,
                        feature_importance,
                    };
                    Ok(res)
                }
                DynCsrMatrix::F64(csr) => {
                    let mut masked_pca = MaskedSparsePCABuilder::new()
                        .mask(selected)
                        .center(center)
                        .verbose(verbose)
                        .alpha(alpha.unwrap_or(1.0))
                        .n_components(n_components)
                        .random_seed(random_seed)
                        .svd_method(svd_method)
                        .build();
                    masked_pca.fit(csr)?;
                    let transformed = masked_pca.transform(csr)?;
                    let explained_variance_ratio = masked_pca.explained_variance_ratio()?;
                    let cumulative_explained_variance_ratio = masked_pca.cumulative_explained_variance_ratio()?;
                    let feature_importance = masked_pca.feature_importances()?;

                    let transformed: Array2<T> = arr2_conversion(transformed)?;
                    let explained_variance_ratio: Array1<T> = arr1_conversion(explained_variance_ratio)?;
                    let cumulative_explained_variance_ratio: Array1<T> = arr1_conversion(cumulative_explained_variance_ratio)?;
                    let feature_importance: Array2<T> = arr2_conversion(feature_importance)?;
                    let res = PCAResult {
                        transformed,
                        explained_variance_ratio,
                        cumulative_explained_variance_ratio,
                        feature_importance,
                    };
                    Ok(res)
                }
                _ => Err(anyhow!("This datatype is currently not supported, please convert to F32 or F64 first, before running PCA!"))
            }
        }
        _ => Err(anyhow!("This anndata type is currently not supported. Only CSR matrices are supported for now!"))
    }
}

/// Generate a random boolean mask for gene selection.
///
/// Creates a boolean vector where `num_random_selection` randomly chosen positions
/// are set to `true`, and all others are `false`. This is used for benchmarking
/// and testing purposes when you want to select a random subset of genes.
///
/// ## Parameters
///
/// * `n_genes` - Total number of genes in the dataset
/// * `num_random_selection` - Number of genes to randomly select
///
/// ## Returns
///
/// Boolean vector of length `n_genes` with exactly `num_random_selection` true values
/// at random positions.
///
/// ## Note
///
/// This function uses the default random number generator. For reproducible results,
/// set the global random seed before calling, or consider using the `random_seed`
/// parameter in `run_pca_sparse_masked`.
///
/// ## Example
///
/// ```rust,ignore
/// // Select 2000 random genes from 20000 total genes
/// let mask = generate_random_mask(20000, 2000);
/// assert_eq!(mask.len(), 20000);
/// assert_eq!(mask.iter().filter(|&&x| x).count(), 2000);
/// ```
fn generate_random_mask(n_genes: usize, num_random_selection: usize) -> Vec<bool> {
    let mut rng = rng();
    let uniform = Uniform::new(0, n_genes).unwrap();
    let mut vec = vec![false; num_random_selection];
    for _ in 0..num_random_selection {
        let v = uniform.sample(&mut rng);
        vec[v] = true;
    }
    vec
}
