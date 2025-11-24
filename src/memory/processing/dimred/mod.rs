//! # Dimensionality Reduction Module
//!
//! This module provides high-performance dimensionality reduction algorithms for single-cell RNA-seq data.
//! Dimensionality reduction is essential for visualization, clustering, and removing noise from high-dimensional
//! single-cell expression data.
//!
//!
//! This module is actively being developed. Currently implemented:
//! - **PCA**: Principal Component Analysis (basic implementation)
//! - **Feature Selection**: Methods for selecting genes/features before dimensionality reduction
//!
//! ## Feature Selection Integration
//!
//! All dimensionality reduction methods support intelligent feature selection to improve
//! performance and biological interpretability:
//!
//! - **Highly Variable Genes**: Use genes with high biological variability
//! - **Full Feature Set**: Use all available genes (not recommended for large datasets)
//! - **Random Selection**: For benchmarking and testing purposes
//!
//! ## Usage Philosophy
//!
//! This module follows single-cell analysis best practices:
//!
//! 1. **Feature Selection First**: Always consider which genes to include
//! 2. **Multiple Methods**: Different algorithms reveal different aspects of data structure
//! 3. **Parameter Tuning**: Most methods require careful parameter selection
//! 4. **Computational Efficiency**: Optimized for large single-cell datasets
//!
//! ## Example Workflow (Planned)
//!
//! ```rust,ignore
//! use single_rust::memory::processing::dimred::{pca, umap, FeatureSelectionMethod};
//!
//! // 1. Select highly variable genes for dimensionality reduction
//! let hvg_mask = compute_highly_variable_genes(&adata, None)?;
//! let feature_selection = FeatureSelectionMethod::HighlyVariableSelection(hvg_mask);
//!
//! // 2. Perform PCA for initial dimensionality reduction
//! pca::run_pca(&adata, Some(50), Some(feature_selection.clone()), Some("X_pca"))?;
//!
//! // 3. Use PCA results for UMAP
//! umap::run_umap(&adata, Some("X_pca"), Some(15), Some("X_umap"))?;
//! ```
//!
//! ## Performance Considerations
//!
//! - **Memory Efficiency**: Supports both dense and sparse matrix operations
//! - **Parallel Computing**: Leverages Rust's concurrency for multi-core processing
//! - **Incremental Algorithms**: Some methods support out-of-core computation for very large datasets
//! - **GPU Acceleration**: Planned integration with GPU compute libraries

pub mod pca;
//pub mod tsne;

/// Methods for selecting features (genes) before dimensionality reduction.
///
/// Feature selection is crucial for dimensionality reduction in single-cell data because:
/// - Reduces computational cost and memory usage
/// - Removes noise from lowly expressed or invariant genes  
/// - Focuses analysis on biologically relevant variability
/// - Improves clustering and visualization quality
///
/// ## Selection Strategies
///
/// - **HighlyVariableSelection**: Use genes identified as highly variable (recommended)
/// - **FullFeatures**: Use all available genes (may include noise)
/// - **RandomSelection**: Random subset of genes (useful for benchmarking)
///
/// ## Usage
///
/// ```rust,ignore
/// // Use highly variable genes (recommended)
/// let hvg_mask = compute_highly_variable_genes(&adata, None)?;
/// let feature_selection = FeatureSelectionMethod::HighlyVariableSelection(hvg_mask);
///
/// // Use all features (not recommended for large datasets)
/// let feature_selection = FeatureSelectionMethod::FullFeatures;
///
/// // Use random subset (for testing/benchmarking)
/// let feature_selection = FeatureSelectionMethod::RandomSelection(2000);
/// ```
#[derive(Clone, Debug)]
pub enum FeatureSelectionMethod {
    /// Use all available genes/features. May include noise and increase computational cost.
    FullFeatures,

    /// Use genes marked as highly variable. Vector of booleans where true indicates
    /// the gene should be included. Length must match the number of genes in the dataset.
    HighlyVariableSelection(Vec<bool>),

    /// Randomly select the specified number of genes. Useful for benchmarking
    /// and testing different feature set sizes.
    RandomSelection(usize),
}
