Quickstart
=========================================

This guide will walk you through a simple example of using SingleRust to analyze single-cell data.

Loading Data
----------------------------

SingleRust supports loading data from H5AD files, which is the standard file format used by the Python package `AnnData <https://anndata.readthedocs.io/>`_.

.. code-block:: rust

   use single_rust::io::{read_h5ad_memory, FileScope};
   use anyhow::Result;

   fn main() -> Result<()> {
       // Load a dataset into memory
       let adata = read_h5ad_memory("path/to/your/data.h5ad")?;

       // Print dataset dimensions
       println!("Dataset dimensions: {} cells x {} genes", adata.n_obs(), adata.n_vars());

       Ok(())
   }

Basic Preprocessing
----------------------------

SingleRust provides functions for common preprocessing steps in single-cell analysis.

Filtering Cells and Genes
~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use single_rust::memory::processing::filtering::{mark_filter_cells, mark_filter_genes};

   // Filter cells based on gene counts and UMI counts
   let cell_filter = mark_filter_cells::<u32, f64>(
       &adata,
       Some(200),   // min_genes: keep cells with at least 200 genes
       Some(5000),  // max_genes: remove cells with more than 5000 genes
       Some(500.0), // min_counts: keep cells with at least 500 UMIs
       None,        // max_counts: no upper limit
       None,        // min_fraction: no minimum fraction
       None,        // max_fraction: no maximum fraction
   )?;

   // Filter genes based on cell counts
   let gene_filter = mark_filter_genes::<u32, f64>(
       &adata,
       Some(10),    // min_cells: keep genes expressed in at least 10 cells
       None,        // max_cells: no upper limit
       None,        // min_counts: no minimum counts
       None,        // max_counts: no maximum counts
       None,        // min_fraction: no minimum fraction
       None,        // max_fraction: no maximum fraction
   )?;

Normalization
~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use single_rust::memory::processing::normalize_expression;
   use single_rust::Direction;
   use single_rust::shared::Precision;

   // Normalize expression by cell (row-wise) to a target sum of 10,000
   normalize_expression(
       &adata.x(),
       10000,               // target sum for each cell
       &Direction::ROW,    // normalize across rows (cells)
       Some(Precision::Single),  // use single precision (f32)
   )?;

   // Log-transform the data
   use single_rust::memory::processing::log1p_expression;

   log1p_expression(&adata.x(), Some(Precision::Single))?;

Finding Highly Variable Genes
~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use single_rust::memory::processing::compute_highly_variable_genes;
   use single_rust::shared::processing::{HVGParams, FlavorType};

   // Find highly variable genes using the Seurat method
   let hvg_params = HVGParams {
       min_mean: 0.0125,
       max_mean: 3.0,
       min_dispersion: 0.5,
       max_dispersion: f64::INFINITY,
       n_bins: 20,
       n_top_genes: Some(2000),  // Select top 2000 genes
       flavor: FlavorType::Seurat,
       span: 0.3,
       batch_key: None,
   };

   compute_highly_variable_genes(&adata, Some(hvg_params))?;


Complete Example
----------------------------

Here's a complete example that loads a dataset, preprocesses it, and generates a PCA plot:

.. code-block:: rust

   use single_rust::io::read_h5ad_memory;
   use single_rust::memory::processing::{
       filtering::{mark_filter_cells, mark_filter_genes},
       normalize_expression, log1p_expression, compute_highly_variable_genes
   };
   use single_rust::memory::plot::plot_pca;
   use single_rust::{Direction, PcaPlotSettings};
   use single_rust::shared::Precision;
   use single_rust::shared::processing::{HVGParams, FlavorType};
   use plotters::style::RGBAColor;
   use anyhow::Result;

   fn main() -> Result<()> {
       // Load dataset
       let adata = read_h5ad_memory("data/sample.h5ad")?;
       println!("Dataset dimensions: {} cells x {} genes", adata.n_obs(), adata.n_vars());

       // Filter cells and genes
       let _cell_filter = mark_filter_cells::<u32, f64>(
           &adata, Some(200), Some(5000), Some(500.0), None, None, None
       )?;

       let _gene_filter = mark_filter_genes::<u32, f64>(
           &adata, Some(10), None, None, None, None, None
       )?;

       // Normalize and log-transform
       normalize_expression(&adata.x(), 10000, &Direction::ROW, Some(Precision::Single))?;
       log1p_expression(&adata.x(), Some(Precision::Single))?;

       // Find highly variable genes
       let hvg_params = HVGParams {
           n_top_genes: Some(2000),
           flavor: FlavorType::Seurat,
           ..Default::default()
       };

       compute_highly_variable_genes(&adata, Some(hvg_params))?;

       println!("Analysis completed successfully!");
       Ok(())
   }

What's Next?
----------------------------

Now that you're familiar with the basics of SingleRust, check out the :doc:`tutorials/index` for more in-depth examples and the :doc:`api/core` for detailed API documentation.