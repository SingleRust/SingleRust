# Single Rust 🧬

## Introduction

Welcome to Single Rust 🚀, a pioneering library for the Rust programming language, focused on the future of production-grade, high-throughput analysis pipelines for single-cell data. SingleRust leverages Rust's fearless concurrency model to transition single-cell data analysis from initial prototyping to robust, scalable deployments.

## Current Status 🚧

SingleRust is in active development with substantial core functionality implemented:

- **Quality Control**: Comprehensive QC metrics with mitochondrial gene detection
- **Normalization**: Log1p and total count normalization with type-safe conversions
- **Feature Selection**: Highly variable gene detection (Seurat, SVR methods)
- **Differential Expression**: Statistical testing (t-tests, Mann-Whitney) with multiple testing correction
- **Dimensionality Reduction**: PCA with sparse matrix support (t-SNE, UMAP in development)
- **Matrix Handling**: Efficient sparse matrix processing and type conversions

## Features 🌟

### Core Functionality

- **AnnData Compatible**: Built on the AnnData Rust ecosystem for seamless data interchange
    - **Note**: There are currently some compatibility limitations with ndarray matrices and rec-arrays (numpy) which are being addressed
- **Backed and In-Memory Processing**: Support for both in-memory and disk-backed operations
- **Type Safety**: Leveraging Rust's type system for robust data analysis

### Performance

- **Fearless Concurrency**: Utilizing Rust's concurrency model for safe, efficient parallel data processing
- **Memory Efficiency**: Optimal memory usage for handling large datasets
- **Sparse Representation**: Specialized handling of sparse data structures common in single-cell data

### Analysis Pipeline

- **Quality Control**: QC metrics, mitochondrial analysis, dropout rates
- **Feature Selection**: Multiple HVG detection methods (Seurat, SVR)
- **Differential Expression**: t-tests, Mann-Whitney U, multiple testing corrections
- **Dimensionality Reduction**: PCA with feature selection integration

## Getting Started 🚀

### Installation

Add SingleRust to your Cargo.toml:

```toml
[dependencies]
single_rust = "0.5.6"
```

### Basic Usage

```rust
use single_rust::io;
use single_rust::memory::processing::{normalize_expression, log1p_expression};
use single_rust::memory::processing::hvg::compute_highly_variable_genes;
use single_rust::memory::processing::diffexp::{rank_gene_groups, CorrectionMethod};
use single_rust::memory::statistics::qc::qc_metrics;
use single_utilities::types::Direction;

// Load data and run complete analysis pipeline
let adata = io::read_h5ad_memory("path/to/data.h5ad")?;

qc_metrics(&adata)?;  // Quality control
log1p_expression(&adata.x(), None)?;  // Log1p normalization
normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;  // Normalize to 10k
compute_highly_variable_genes(&adata, None)?;  // Find HVGs

// Differential expression analysis
rank_gene_groups(&adata, "cell_type", Some("rest"), None, None, None, 
                Some(100), CorrectionMethod::BejaminiHochberg, None, None)?;
```

## Differential Expression Analysis 🧪

Robust statistical testing for identifying differentially expressed genes between cell populations:

- **Statistical tests**: t-tests (Student's, Welch's), Mann-Whitney U test
- **Multiple testing correction**: Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli
- **Effect sizes**: Log fold changes with configurable pseudocounts
- **Flexible comparisons**: Group vs group, group vs rest, or custom comparisons

```rust
// Compare cell types using Mann-Whitney test
use single_statistics::testing::TestMethod;

rank_gene_groups(&adata, "cell_type", Some("T_cells"), Some(&["B_cells"]), 
                Some("comparison"), Some(TestMethod::MannWhitney), Some(50),
                CorrectionMethod::BejaminiHochberg, Some(true), Some(1.0))?;
```

## Quality Control & Analysis Features 🔍

- **Quality Control**: Automatic mitochondrial gene detection, cell/gene metrics, dropout analysis
- **Highly Variable Genes**: Seurat and SVR methods with flexible selection criteria  
- **Dimensionality Reduction**: PCA with sparse matrix support, t-SNE/UMAP in development

```rust
// Complete QC and feature selection workflow
use single_rust::shared::HVGParams;

qc_metrics(&adata)?;
compute_highly_variable_genes(&adata, Some(HVGParams { 
    n_top_genes: Some(2000), ..Default::default() 
}))?;

// PCA with HVG feature selection
use single_rust::memory::processing::dimred::{FeatureSelectionMethod, pca::run_pca_sparse_masked};

let pca_result = run_pca_sparse_masked::<f64>(&adata.x(), 
    Some(FeatureSelectionMethod::HighlyVariableSelection(hvg_mask)), 
    Some(true), None, Some(50), None, Some(42), None)?;
```

## Visualization Strategy 📊

Rather than implementing visualization directly in Rust, SingleRust focuses on computation while enabling visualization through data exports:

- **External Tool Integration**: Export functions (in development) will allow seamless integration with Python and R visualization libraries
- **Familiar Plotting**: Users can continue using their preferred plotting tools in Python and R
- **Performance Balance**: Computationally intensive analysis in Rust with visualization in languages with mature plotting libraries
- **Export Formats**: CSV and other standard formats for maximum compatibility

This approach combines Rust's performance benefits for computation with the rich visualization ecosystems of Python and R.

## Roadmap 🗺️

- **Near-term**: t-SNE/UMAP, clustering algorithms (Leiden, k-means), enhanced Python/R export
- **Medium-term**: Trajectory analysis, batch correction, spatial transcriptomics support  
- **Long-term**: Complete pipeline integration, web interface, cloud scaling

## Documentation 📚

Comprehensive documentation with API docs (`cargo doc --open`) and scientific context for all modules.

## Contributing 🤝

Contributions welcome! Areas needing development: algorithm implementations, performance optimization, testing, and documentation. See issues for specific needs.

## License 📜

SingleRust is distributed under the BSD 3-Clause License, ensuring it remains free and open for all to use and contribute to.

## Contact 📧

For inquiries, suggestions, or expressions of interest in contributing, please open an issue on our GitHub repository or reach out directly via [email](info@single-rust.com).

## Acknowledgements 🙏

- The Rust Community: For providing an inspiring example of what open-source collaboration can achieve.
- The single-cell bioinformatics community: For developing innovative algorithms and approaches.

## IMPORTANT 🚨

This library is still in active development and highly unoptimized in some areas. If you want to contribute, please go for it! We especially welcome help in performance optimization, test coverage, and documentation.