# Single Rust 🧬

## Introduction

Welcome to Single Rust 🚀, a pioneering library for the Rust programming language, focused on the future of production-grade, high-throughput analysis pipelines for single-cell data. SingleRust leverages Rust's fearless concurrency model to transition single-cell data analysis from initial prototyping to robust, scalable deployments.

## Current Status 🚧

SingleRust is currently in active development, with core functionality already implemented:

- **Matrix Handling**: Efficient processing of sparse matrices common in single-cell data
- **Quality Control**: Tools for filtering cells and genes based on expression metrics
- **Normalization**: Implementation of standard normalization procedures
- **Highly Variable Gene Detection**: Algorithms for identifying genes with high variability
- **Core Statistics**: Fast computation of essential statistics for single-cell analysis

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

- **Quality Control**: Tools for filtering cells and genes based on expression metrics
- **Normalization**: Standard normalization procedures for single-cell data
- **Feature Selection**: Identification of highly variable genes for dimensionality reduction
- **Differential Expression**: (Coming soon) Tools for identifying differentially expressed genes between cell populations

## Getting Started 🚀

### Installation

Add SingleRust to your Cargo.toml:

```toml
[dependencies]
single_rust = "0.2.1-alpha.1"
```

### Basic Usage

```rust
use single_rust::io;
use single_rust::memory::processing::{normalize_expression, log1p_expression};
use single_rust::shared::Direction;

// Load an AnnData file into memory
let adata = io::read_h5ad_memory("path/to/data.h5ad")?;

// Perform log1p normalization
log1p_expression(&adata.x(), None)?;

// Normalize expression (e.g., to 10,000 counts per cell)
normalize_expression(&adata.x(), 10_000, &Direction::ROW, None)?;

// Compute highly variable genes
use single_rust::memory::processing::compute_highly_variable_genes;
compute_highly_variable_genes(&adata, None)?;
```

## Differential Expression Analysis 🧪

Differential expression analysis in SingleRust is designed to efficiently identify genes that show significant differences between cell populations. The implementation focuses on:

- **Statistical Robustness**: Implementation of well-established statistical tests
- **Performance**: Optimized for large single-cell datasets
- **Flexibility**: Support for various experimental designs and comparison strategies

Already implemented features include:
- Rank-based tests (Wilcoxon)
- Parametric tests (t-test)
- Multiple testing correction
- Effect size calculation

This module is designed with computational efficiency in mind, focusing on the statistics rather than visualization, allowing it to handle large datasets with low memory footprint.

## Visualization Strategy 📊

Rather than implementing visualization directly in Rust, SingleRust focuses on computation while enabling visualization through data exports:

- **External Tool Integration**: Export functions (in development) will allow seamless integration with Python and R visualization libraries
- **Familiar Plotting**: Users can continue using their preferred plotting tools in Python and R
- **Performance Balance**: Computationally intensive analysis in Rust with visualization in languages with mature plotting libraries
- **Export Formats**: CSV and other standard formats for maximum compatibility

This approach combines Rust's performance benefits for computation with the rich visualization ecosystems of Python and R.

## Roadmap 🗺️

- **Dimensionality Reduction**: PCA, t-SNE, and UMAP implementations
- **Clustering**: Graph-based and k-means clustering algorithms
- **Advanced Trajectory Analysis**: Tools for pseudotime and lineage inference
- **Integration Methods**: Batch correction and dataset integration
- **Spatial Applications**: Analysis of spatial transcriptomics data
- **Export Functions**: Tools for exporting analysis results to formats compatible with visualization libraries in Python and R
- **Full ndarray and rec-array Compatibility**: Complete interoperability with numpy array formats

## Contributing 🤝

We welcome contributions from the community! Whether it's adding new features, improving documentation, or reporting bugs, your help is appreciated.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please check the issue tracker for areas where help is needed.

## License 📜

SingleRust is distributed under the BSD 3-Clause License, ensuring it remains free and open for all to use and contribute to.

## Contact 📧

For inquiries, suggestions, or expressions of interest in contributing, please open an issue on our GitHub repository or reach out directly via [email](single-rust@crimelabs.eu).

## Acknowledgements 🙏

- The Rust Community: For providing an inspiring example of what open-source collaboration can achieve.
- The single-cell bioinformatics community: For developing innovative algorithms and approaches.

## IMPORTANT 🚨

This library is still in active development and highly unoptimized in some areas. If you want to contribute, please go for it! We especially welcome help in performance optimization, test coverage, and documentation.