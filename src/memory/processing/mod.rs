//! # Memory Processing Module
//!
//! This module provides high-performance, in-memory processing functions for single-cell RNA-seq data analysis.
//! All functions operate on `IMAnnData` objects and their associated expression matrices.
//!
//! ## Modules
//!
//! - **`diffexp`**: Differential expression analysis with statistical tests (t-test, Mann-Whitney, Wilcoxon)
//! - **`dimred`**: Dimensionality reduction algorithms (PCA, t-SNE, UMAP) - *In development*
//! - **`filtering`**: Cell and gene filtering based on expression criteria
//! - **`hvg`**: Highly variable gene detection algorithms
//! - **`transformation`**: Expression data transformations (normalization, log1p)
pub mod diffexp;
pub mod dimred;
pub mod filtering;
mod hvg;
mod transformation;

mod enrichment;

pub use hvg::compute_highly_variable_genes;
pub use transformation::log1p_expression;
pub use transformation::normalize_expression;
