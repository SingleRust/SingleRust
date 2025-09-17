//! # Quality Control Metrics for Single-Cell Data
//!
//! This module provides comprehensive quality control (QC) analysis for single-cell RNA-seq datasets.
//! QC metrics are essential for identifying low-quality cells and problematic genes that should be
//! filtered before downstream analysis.
//!
//! ## Why Quality Control?
//!
//! Single-cell RNA-seq data contains various sources of technical variation and artifacts:
//! - **Low-quality cells**: Dying cells, doublets, empty droplets
//! - **Technical noise**: PCR amplification bias, sequencing depth variation
//! - **Biological confounders**: Cell cycle, stress responses, batch effects
//! - **Gene-specific issues**: Dropout, low detection rates
//!
//! ## Key QC Metrics
//!
//! ### Cell-level Metrics (Observations)
//! - **Total counts**: Total UMI/read count per cell
//! - **Number of genes**: Count of detected genes per cell
//! - **Mitochondrial percentage**: Fraction of reads from mitochondrial genes
//! - **Top gene percentages**: Expression concentration in highly expressed genes
//!
//! ### Gene-level Metrics (Variables)  
//! - **Total counts**: Total expression across all cells
//! - **Mean expression**: Average expression per gene
//! - **Detection rate**: Percentage of cells expressing each gene
//! - **Dropout percentage**: Fraction of cells with zero expression
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use single_rust::memory::statistics::qc::{calculate_qc_metrics, qc_metrics};
//!
//! // Quick QC with mitochondrial genes (recommended for most datasets)
//! qc_metrics(&adata)?;
//!
//! // Custom QC analysis
//! let (obs_qc, var_qc) = calculate_qc_metrics(
//!     &adata,
//!     Some("counts"),              // Expression type name
//!     Some("genes"),               // Variable type name  
//!     Some(vec!["mito", "ribo"]),  // QC gene sets to analyze
//!     Some(vec![20, 50, 100]),     // Top N genes for concentration analysis
//!     None,                        // Use main expression matrix
//!     false,                       // Don't use raw data
//!     true,                        // Store results in adata
//!     true,                        // Include log1p transformations
//! )?.unwrap();
//! ```
//!
//! ## Typical QC Workflow
//!
//! 1. **Calculate metrics**: Run QC analysis on raw count data
//! 2. **Visualize distributions**: Plot histograms of key metrics
//! 3. **Set thresholds**: Determine filtering criteria based on distributions
//! 4. **Filter data**: Remove low-quality cells and poorly detected genes
//! 5. **Re-evaluate**: Check that filtering improved data quality
//!
//! ## Common QC Thresholds
//!
//! ### Cell Filtering (typical ranges)
//! - **Min genes per cell**: 200-500 genes
//! - **Max genes per cell**: 2500-5000 genes (removes doublets)
//! - **Min UMI per cell**: 500-1000 counts
//! - **Max mitochondrial %**: 15-25% (removes dying cells)
//!
//! ### Gene Filtering
//! - **Min cells expressing**: 3-10 cells (removes rarely expressed genes)
//! - **Min total counts**: 10-50 counts across all cells
//!
//! ## Output Columns
//!
//! ### Added to `adata.obs` (cell-level)
//! - `n_genes_by_counts`: Number of genes with non-zero counts
//! - `total_counts`: Total UMI/read count per cell
//! - `pct_counts_mito`: Percentage of mitochondrial gene expression
//! - `pct_counts_in_top_N_genes`: Expression concentration metrics
//! - Log1p versions of count metrics (if enabled)
//!
//! ### Added to `adata.var` (gene-level)  
//! - `n_cells_by_counts`: Number of cells expressing each gene
//! - `mean_counts`: Mean expression per gene
//! - `total_counts`: Total expression across all cells
//! - `pct_dropout_by_counts`: Percentage of cells with zero expression
//! - Log1p versions of count metrics (if enabled)

use anndata_memory::{IMAnnData, IMArrayElement};
use num_traits::Float;
use polars::{frame::DataFrame, prelude::Column};
use single_utilities::types::Direction;

use crate::{ComputeNonZero, ComputeSum};

/// Calculate cell-level (observation) quality control metrics.
///
/// Computes comprehensive QC statistics for each cell including expression totals,
/// gene detection rates, and specialized metrics like mitochondrial gene percentages.
///
/// ## Parameters
/// * `adata` - AnnData object containing the dataset
/// * `x` - Expression matrix to analyze
/// * `expr_type` - Label for expression type (e.g., "counts", "UMI")
/// * `var_type` - Label for variable type (e.g., "genes", "features")
/// * `qc_vars` - Gene sets to analyze (e.g., mitochondrial, ribosomal)
/// * `percent_top` - Top N genes for concentration analysis
/// * `log1p` - Whether to include log1p-transformed metrics
///
/// ## Returns
/// DataFrame with cell-level QC metrics as columns
///
/// ## Metrics Computed
/// - Total expression per cell
/// - Number of detected genes per cell  
/// - Percentage of expression from specific gene sets
/// - Expression concentration in top N genes
/// - Optional log1p transformations for all count metrics
fn describe_obs(
    adata: &IMAnnData,
    x: &IMArrayElement,
    expr_type: &str,
    var_type: &str,
    qc_vars: &[&str],
    percent_top: &[usize],
    log1p: bool,
) -> anyhow::Result<DataFrame> {
    let n_obs = adata.n_obs();
    let n_vars = adata.n_vars();

    let n_genes_by_counts: Vec<u32> = x.nonzero_whole(&Direction::ROW)?;
    let total_counts: Vec<f64> = x.sum_whole(&Direction::ROW)?;

    let mut columns = vec![];

    let col_name = format!("n_{}_by_{}", var_type, expr_type);
    columns.push(Column::new(col_name.into(), n_genes_by_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = n_genes_by_counts
            .iter()
            .map(|&x| (x as f64 + 1.0).ln())
            .collect();
        let col_name = format!("log1p_n_{}_by_{}", var_type, expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    let col_name = format!("total_{}", expr_type);
    columns.push(Column::new(col_name.into(), total_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = total_counts.iter().map(|&x| (x + 1.0).ln()).collect();
        let col_name = format!("log1p_total_{}", expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    if !percent_top.is_empty() {
        use crate::shared::statistics::ComputeTopSegmentProportions;
        let proportions = x.top_segment_proportions(&Direction::ROW, percent_top)?;
        for (i, &n) in percent_top.iter().enumerate() {
            let values: Vec<f64> = proportions.column(i).iter().map(|&x| x * 100.0).collect();
            let col_name = format!("pct_{}_in_top_{}_{}", expr_type, n, var_type);
            columns.push(Column::new(col_name.into(), values));
        }
    }

    for qc_var in qc_vars {
        let var_mask = adata
            .var()
            .get_column_from_df(qc_var)?
            .bool()?
            .into_iter()
            .map(|x| x.unwrap_or(false))
            .collect::<Vec<bool>>();

        let qc_var_indices: Vec<usize> = var_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &mask)| if mask { Some(i) } else { None })
            .collect();

        let qc_totals = if !qc_var_indices.is_empty() {
            let mut mask = vec![false; n_vars];
            for &idx in &qc_var_indices {
                if idx < n_vars {
                    mask[idx] = true;
                }
            }

            x.sum_whole_masked(&Direction::ROW, &mask)?
        } else {
            vec![0.0; n_obs]
        };

        let col_name = format!("total_{}_{}", expr_type, qc_var);
        columns.push(Column::new(col_name.into(), qc_totals.clone()));

        if log1p {
            let log_values: Vec<f64> = qc_totals.iter().map(|&x| (x + 1.0).ln()).collect();
            let col_name = format!("log1p_total_{}_{}", expr_type, qc_var);
            columns.push(Column::new(col_name.into(), log_values));
        }

        let pct_values: Vec<f64> = qc_totals
            .iter()
            .zip(total_counts.iter())
            .map(|(&qc, &total)| if total > 0.0 { qc / total * 100.0 } else { 0.0 })
            .collect();
        let col_name = format!("pct_{}_{}", expr_type, qc_var);
        columns.push(Column::new(col_name.into(), pct_values));
    }

    DataFrame::new(columns).map_err(Into::into)
}

/// Calculate gene-level (variable) quality control metrics.
///
/// Computes QC statistics for each gene including detection rates across cells,
/// mean expression levels, and dropout percentages.
///
/// ## Parameters
/// * `adata` - AnnData object containing the dataset
/// * `x` - Expression matrix to analyze
/// * `expr_type` - Label for expression type (e.g., "counts", "UMI")
/// * `_var_type` - Label for variable type (unused, kept for API consistency)
/// * `log1p` - Whether to include log1p-transformed metrics
///
/// ## Returns
/// DataFrame with gene-level QC metrics as columns
///
/// ## Metrics Computed
/// - Number of cells expressing each gene
/// - Mean expression per gene across all cells
/// - Total expression per gene across all cells
/// - Dropout percentage (fraction of cells with zero expression)
/// - Optional log1p transformations for count metrics
///
/// ## Use Cases
/// - Identify poorly detected genes for filtering
/// - Assess gene expression distributions
/// - Quality control before feature selection
fn describe_var(
    adata: &IMAnnData,
    x: &anndata_memory::IMArrayElement,
    expr_type: &str,
    _var_type: &str,
    log1p: bool,
) -> anyhow::Result<DataFrame> {
    let n_obs = adata.n_obs();

    let n_cells_by_counts: Vec<u32> = x.nonzero_whole(&Direction::COLUMN)?;
    let total_counts: Vec<f64> = x.sum_whole(&Direction::COLUMN)?;

    let mean_counts: Vec<f64> = total_counts
        .iter()
        .map(|&total| total / n_obs as f64)
        .collect();

    let mut columns = vec![];

    let col_name = format!("n_cells_by_{}", expr_type);
    columns.push(Column::new(col_name.into(), n_cells_by_counts.clone()));

    let col_name = format!("mean_{}", expr_type);
    columns.push(Column::new(col_name.into(), mean_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = mean_counts.iter().map(|&x| (x + 1.0).ln()).collect();
        let col_name = format!("log1p_mean_{}", expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    let pct_dropout: Vec<f64> = n_cells_by_counts
        .iter()
        .map(|&n| (1.0 - n as f64 / n_obs as f64) * 100.0)
        .collect();
    let col_name = format!("pct_dropout_by_{}", expr_type);
    columns.push(Column::new(col_name.into(), pct_dropout));

    let col_name = format!("total_{}", expr_type);
    columns.push(Column::new(col_name.into(), total_counts.clone()));

    if log1p {
        let log_values: Vec<f64> = total_counts.iter().map(|&x| (x + 1.0).ln()).collect();
        let col_name = format!("log1p_total_{}", expr_type);
        columns.push(Column::new(col_name.into(), log_values));
    }

    DataFrame::new(columns).map_err(Into::into)
}

/// Calculate comprehensive quality control metrics for single-cell data.
///
/// This is the main function for QC analysis, computing both cell-level and gene-level
/// metrics that are essential for identifying low-quality data and setting filtering thresholds.
///
/// ## Parameters
///
/// * `adata` - AnnData object containing expression data
/// * `expr_type` - Name for expression data type (default: "counts")
/// * `var_type` - Name for variable type (default: "genes")  
/// * `qc_vars` - Gene sets to analyze (e.g., mitochondrial genes)
/// * `percent_top` - Top N genes for concentration analysis (default: [50,100,200,500])
/// * `layer` - Specific layer to analyze (None = main matrix)
/// * `use_raw` - Use raw data if available (not yet implemented)
/// * `inplace` - Store results in adata (true) or return DataFrames (false)
/// * `log1p` - Include log1p-transformed versions of count metrics
///
/// ## Returns
///
/// - If `inplace=true`: Results stored in `adata.obs` and `adata.var`, returns `None`
/// - If `inplace=false`: Returns `Some((obs_metrics, var_metrics))` DataFrames
///
/// ## Usage Examples
///
/// ```rust,ignore
/// // Store results directly in adata
/// calculate_qc_metrics(&adata, None, None, Some(vec!["mito"]), None, None, false, true, true)?;
///
/// // Get results as separate DataFrames
/// let (obs_qc, var_qc) = calculate_qc_metrics(
///     &adata, None, None, None, None, None, false, false, true
/// )?.unwrap();
/// ```
///
/// ## Quality Control Gene Sets
///
/// Common gene sets for QC analysis:
/// - **Mitochondrial**: Genes starting with "MT-" or "mt-"
/// - **Ribosomal**: Genes starting with "RPS" or "RPL"  
/// - **Hemoglobin**: Genes like "HBA1", "HBB" for blood contamination
/// - **Heat shock**: Stress response genes
///
/// ## Typical Workflow
///
/// 1. Run QC analysis with mitochondrial genes
/// 2. Plot distributions of key metrics
/// 3. Set filtering thresholds based on data characteristics
/// 4. Apply filters to remove low-quality cells/genes
/// 5. Re-run QC to verify filtering effectiveness
#[allow(clippy::too_many_arguments)]
pub fn calculate_qc_metrics(
    adata: &IMAnnData,
    expr_type: Option<&str>,
    var_type: Option<&str>,
    qc_vars: Option<Vec<&str>>,
    percent_top: Option<Vec<usize>>,
    layer: Option<&str>,
    use_raw: bool,
    inplace: bool,
    log1p: bool,
) -> anyhow::Result<Option<(DataFrame, DataFrame)>> {
    let expr_type = expr_type.unwrap_or("counts");
    let var_type = var_type.unwrap_or("genes");
    let qc_vars = qc_vars.unwrap_or_default();
    let percent_top = percent_top.unwrap_or_else(|| vec![50, 100, 200, 500]);

    let x = if let Some(layer_name) = layer {
        adata.layers().get_array_shallow(layer_name)?
    } else if use_raw {
        return Err(anyhow::anyhow!("Raw data access not yet implemented"));
    } else {
        adata.x()
    };

    let obs_metrics = describe_obs(
        adata,
        &x,
        expr_type,
        var_type,
        &qc_vars,
        &percent_top,
        log1p,
    )?;

    let var_metrics = describe_var(adata, &x, expr_type, var_type, log1p)?;

    if inplace {
        let mut obs_df = adata.obs().get_data();
        for col in obs_metrics.get_columns() {
            obs_df.with_column(col.clone())?;
        }
        adata.obs().set_data(obs_df)?;

        let mut var_df = adata.var().get_data();
        for col in var_metrics.get_columns() {
            var_df.with_column(col.clone())?;
        }
        adata.var().set_data(var_df)?;

        Ok(None)
    } else {
        Ok(Some((obs_metrics, var_metrics)))
    }
}

/// Convenient function to calculate standard QC metrics with mitochondrial genes.
///
/// This is a simplified interface for the most common QC analysis in single-cell studies.
/// It automatically identifies mitochondrial genes and computes standard QC metrics.
///
/// ## What it does
///
/// 1. **Identifies mitochondrial genes**: Finds genes starting with "MT-" or "mt-"
/// 2. **Adds mito annotation**: Creates boolean column in `adata.var` marking mitochondrial genes
/// 3. **Calculates QC metrics**: Runs comprehensive QC analysis including:
///    - Total counts and gene numbers per cell
///    - Mitochondrial gene percentage per cell
///    - Top gene concentration metrics (top 50, 100, 200, 500 genes)
///    - Gene detection rates and dropout percentages
///    - Log1p transformed versions of all count metrics
///
/// ## Parameters
/// * `adata` - AnnData object to analyze
///
/// ## Results
/// All results are stored directly in `adata`:
/// - Cell-level metrics added to `adata.obs`
/// - Gene-level metrics added to `adata.var`
/// - Mitochondrial gene annotation in `adata.var["mito"]`
///
/// ## Usage
///
/// ```rust,ignore
/// // Run standard QC analysis
/// qc_metrics(&adata)?;
///
/// // Access results
/// let obs_df = adata.obs().get_data();
/// let mito_pct = obs_df.column("pct_counts_mito")?;
/// let n_genes = obs_df.column("n_genes_by_counts")?;
/// ```
///
/// ## Typical Next Steps
///
/// After running this function, you typically:
/// 1. Plot distributions of `pct_counts_mito`, `n_genes_by_counts`, `total_counts`
/// 2. Set filtering thresholds based on these distributions
/// 3. Filter cells and genes using these thresholds
/// 4. Proceed with normalization and downstream analysis
///
/// ## Mitochondrial Gene Detection
///
/// This function identifies mitochondrial genes using standard naming conventions:
/// - Human: Genes starting with "MT-" (e.g., MT-CO1, MT-ND1)
/// - Mouse: Genes starting with "mt-" (e.g., mt-Co1, mt-Nd1)
///
/// If your data uses different naming conventions, use `calculate_qc_metrics` directly
/// with custom gene sets.
pub fn qc_metrics(adata: &IMAnnData) -> anyhow::Result<()> {
    let var_names = adata.var_names();
    let mito_mask: Vec<bool> = var_names
        .iter()
        .map(|name| name.starts_with("MT-") || name.starts_with("mt-"))
        .collect();

    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("mito".into(), mito_mask))?;
    adata.var().set_data(var_df)?;

    calculate_qc_metrics(
        adata,
        Some("counts"),
        Some("genes"),
        Some(vec!["mito"]),
        Some(vec![50, 100, 200, 500]),
        None,
        false,
        true,
        true,
    )?;

    Ok(())
}
