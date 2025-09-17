//! # Highly Variable Gene Detection
//!
//! This module implements methods for identifying highly variable genes (HVGs) in single-cell RNA-seq data.
//! HVGs are genes that exhibit high variability across cells beyond what would be expected from technical noise,
//! and are crucial for downstream analysis like dimensionality reduction and clustering.
//!
//! ## Why Highly Variable Genes?
//!
//! Single-cell datasets typically contain 20,000+ genes, but most show little biological variation:
//! - **Technical noise**: Sequencing depth, amplification bias, dropout events
//! - **Housekeeping genes**: Consistently expressed across cell types
//! - **Lowly expressed genes**: Dominated by technical variation
//!
//! By focusing on highly variable genes, we:
//! - Reduce computational burden and noise
//! - Highlight biologically relevant variation
//! - Improve clustering and visualization quality
//! - Focus on genes driving cell-to-cell differences
//!
//! ## Implemented Methods
//!
//! ### 1. Seurat Method (Default)
//! Based on Satija lab's approach, models the mean-variance relationship using binning:
//! - Bins genes by expression level
//! - Calculates expected dispersion for each bin
//! - Identifies genes with higher-than-expected dispersion
//! - **Best for**: Most single-cell datasets, widely validated
//!
//! ### 2. SVR Method  
//! Uses Support Vector Regression to model mean-variance relationship:
//! - Fits SVR to log(mean) vs log(variance)
//! - Identifies genes with high residuals from the trend
//! - **Best for**: Datasets with complex mean-variance relationships
//!
//! ### 3. Cell Ranger Method
//! 10x Genomics approach (not yet implemented):
//! - Similar to Seurat but with different normalization
//! - **Best for**: Datasets processed with Cell Ranger pipeline
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use single_rust::memory::processing::hvg::compute_highly_variable_genes;
//! use single_rust::shared::processing::HVGParams;
//!
//! // Basic usage with defaults (Seurat method)
//! compute_highly_variable_genes(&adata, None)?;
//!
//! // Custom parameters
//! let params = HVGParams {
//!     n_top_genes: Some(2000),     // Select top 2000 most variable genes
//!     min_mean: 0.0125,            // Minimum mean expression
//!     max_mean: 3.0,               // Maximum mean expression  
//!     min_dispersion: 0.5,         // Minimum normalized dispersion
//!     n_bins: 20,                  // Number of expression bins
//!     flavor: FlavorType::Seurat,  // Method to use
//! };
//! compute_highly_variable_genes(&adata, Some(params))?;
//!
//! // Access results
//! let var_df = adata.var().get_data();
//! let hvg_mask = var_df.column("highly_variable")?.bool()?;
//! let hvg_genes: Vec<_> = hvg_mask.into_iter()
//!     .enumerate()
//!     .filter_map(|(i, is_hvg)| if is_hvg.unwrap_or(false) { Some(i) } else { None })
//!     .collect();
//! ```
//!
//! ## Output Columns
//!
//! Results are stored in `adata.var()` with method-specific columns:
//!
//! ### Seurat Method
//! - `means`: Log1p mean expression per gene
//! - `dispersions`: Log dispersion (variance/mean)
//! - `dispersions_norm`: Normalized dispersion scores
//! - `highly_variable`: Boolean mask for selected genes
//!
//! ### SVR Method  
//! - `means`: Mean expression per gene
//! - `variances`: Variance per gene
//! - `residuals`: Residuals from SVR fit
//! - `residuals_standardized`: Standardized residuals
//! - `mean_variance_trend`: Predicted variance from SVR
//! - `highly_variable`: Boolean mask for selected genes
//!
//! ## Parameter Guidelines
//!
//! ### Gene Count Selection
//! - **1000-2000 genes**: Good for initial exploration
//! - **2000-5000 genes**: Standard for most analyses  
//! - **5000+ genes**: May include noise, use with caution
//!
//! ### Expression Filters
//! - `min_mean`: Remove lowly expressed genes (default: 0.0125)
//! - `max_mean`: Remove highly expressed genes (default: 3.0)
//! - Adjust based on your normalization and log-transformation
//!
//! ### Method Selection
//! - **Seurat**: Robust, widely used, good default choice
//! - **SVR**: Better for complex datasets, more computationally intensive
//! - **Cell Ranger**: Use if data processed with 10x Cell Ranger

use crate::shared::processing::{fit_svr, standardize_log_form_vec, FlavorType, HVGParams};
use crate::{ComputeSum, ComputeVariance};
use anndata_memory::{IMAnnData, IMArrayElement};
use polars::prelude::Column;
use single_utilities::types::Direction;

/// Compute highly variable genes using the specified method and parameters.
///
/// This is the main entry point for HVG detection. It analyzes the mean-variance relationship
/// in the expression data to identify genes with biological variability above technical noise.
///
/// ## Algorithm Overview
///
/// 1. **Calculate Statistics**: Compute mean and variance for each gene
/// 2. **Model Relationship**: Fit mean-variance relationship using chosen method
/// 3. **Identify Outliers**: Find genes with higher variability than expected
/// 4. **Apply Filters**: Filter by expression level and dispersion thresholds
/// 5. **Store Results**: Add columns to `adata.var()` with results
///
/// ## Parameters
///
/// * `adata` - AnnData object containing expression data
/// * `params` - HVG parameters (uses defaults if None)
///
/// ## Method Details
///
/// - **Seurat**: Bins genes by expression, normalizes dispersion within bins
/// - **SVR**: Fits support vector regression to model mean-variance trend
/// - **Cell Ranger**: (Not implemented) 10x Genomics approach
///
/// ## Results Storage
///
/// All results are stored as new columns in `adata.var()`. The exact columns
/// depend on the method used, but all methods add a `highly_variable` boolean column.
///
/// ## Examples
///
/// ```rust,ignore
/// // Use defaults (Seurat method, top 2000 genes)
/// compute_highly_variable_genes(&adata, None)?;
///
/// // Custom SVR analysis
/// let params = HVGParams {
///     flavor: FlavorType::SVR,
///     n_top_genes: Some(3000),
///     min_mean: 0.01,
///     max_mean: 4.0,
///     min_dispersion: 0.3,
///     n_bins: 25,
/// };
/// compute_highly_variable_genes(&adata, Some(params))?;
/// ```
pub fn compute_highly_variable_genes(
    adata: &IMAnnData,
    params: Option<HVGParams>,
) -> anyhow::Result<()> {
    let params = params.unwrap_or_default();
    let x = adata.x();

    match params.flavor {
        FlavorType::Seurat => compute_seurat_hvg(adata, &x, params),
        FlavorType::CellRanger => compute_cell_ranger_hvg(adata, &x, params),
        FlavorType::SVR => compute_svr_hvg(adata, &x, params),
    }
}

/// Post-process dispersion statistics for Seurat method to handle edge cases.
///
/// Handles bins with single genes where standard deviation cannot be computed.
/// Following the original Seurat implementation, single-gene bins get special treatment
/// to ensure they can still contribute to HVG selection.
///
/// ## Parameters
/// * `bin_means` - Mean dispersion values for each bin
/// * `bin_stds` - Standard deviation of dispersions for each bin
///
/// ## Behavior
/// For bins with NaN standard deviation (single gene bins):
/// - Sets std = mean and mean = 0
/// - This results in normalized dispersion = 1 for these genes
fn postprocess_seurat_dispersions(
    bin_means: &mut [f64],
    bin_stds: &mut [f64],
) -> anyhow::Result<()> {
    // This matches Python's _postprocess_dispersions_seurat
    for i in 0..bin_means.len() {
        if bin_stds[i].is_nan() {
            // For single-gene bins, set std = mean and mean = 0
            // This effectively sets normalized dispersion to 1
            bin_stds[i] = bin_means[i];
            bin_means[i] = 0.0;
        }
    }
    Ok(())
}

/// Create equal-width bins for gene expression levels.
///
/// Divides the expression range into equal-width bins for the Seurat method.
/// This allows modeling of mean-variance relationship by expression level,
/// as highly expressed genes naturally have different variance characteristics.
///
/// ## Parameters
/// * `log_means` - Log-transformed mean expression values
/// * `n_bins` - Number of bins to create
///
/// ## Returns
/// * Bin assignment for each gene (0-based indices)
/// * Bin edge values for reference
///
/// ## Algorithm
/// 1. Find min/max of valid (finite) expression values
/// 2. Create equal-width intervals between min and max
/// 3. Assign each gene to appropriate bin
/// 4. Handle edge cases (NaN values, exact maximum)
fn equal_width_binning(log_means: &[f64], n_bins: usize) -> anyhow::Result<(Vec<usize>, Vec<f64>)> {
    // Find min and max (excluding NaN)
    let mut valid_means: Vec<f64> = log_means
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect();

    if valid_means.is_empty() {
        return Err(anyhow::anyhow!("No valid mean values found"));
    }

    valid_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_mean = valid_means[0];
    let max_mean = valid_means[valid_means.len() - 1];

    // Create equal-width bins
    let bin_width = (max_mean - min_mean) / n_bins as f64;
    let mut bin_edges = vec![0.0; n_bins + 1];

    for (i, edge) in bin_edges.iter_mut().enumerate().take(n_bins + 1) {
        *edge = min_mean + (i as f64) * bin_width;
    }

    // Make sure the last edge includes the maximum value
    bin_edges[n_bins] = max_mean + 1e-10;

    // Assign each gene to a bin
    let mut bin_indices = vec![0; log_means.len()];

    for (i, &mean) in log_means.iter().enumerate() {
        if !mean.is_finite() {
            bin_indices[i] = 0; // Assign NaN/inf to first bin
            continue;
        }

        // Find which bin this value belongs to
        let mut bin_idx = 0;
        for j in 0..n_bins {
            if mean >= bin_edges[j] && mean < bin_edges[j + 1] {
                bin_idx = j;
                break;
            }
        }

        // Handle edge case where mean == max_mean
        if mean == max_mean {
            bin_idx = n_bins - 1;
        }

        bin_indices[i] = bin_idx;
    }

    Ok((bin_indices, bin_edges))
}

/// Calculate mean and standard deviation of dispersions within each expression bin.
///
/// For the Seurat method, this establishes the expected mean-variance relationship
/// by computing statistics within bins of similar expression levels.
///
/// ## Parameters
/// * `log_dispersions` - Log-transformed dispersion values (variance/mean)
/// * `bin_indices` - Bin assignment for each gene
/// * `n_bins` - Total number of bins
///
/// ## Returns
/// * Mean dispersion for each bin
/// * Standard deviation of dispersion for each bin
///
/// ## Special Cases
/// - Empty bins: Both mean and std set to NaN
/// - Single-gene bins: Mean = gene value, std = NaN (handled by postprocessing)
/// - Uses sample standard deviation (n-1 denominator)
fn calculate_bin_stats(
    log_dispersions: &[f64],
    bin_indices: &[usize],
    n_bins: usize,
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let mut bin_values: Vec<Vec<f64>> = vec![Vec::new(); n_bins];

    // Collect values for each bin (excluding NaN)
    for (i, &bin_idx) in bin_indices.iter().enumerate() {
        let disp = log_dispersions[i];
        if !disp.is_nan() && bin_idx < n_bins {
            bin_values[bin_idx].push(disp);
        }
    }

    let mut bin_means = vec![0.0; n_bins];
    let mut bin_stds = vec![0.0; n_bins];

    for bin_idx in 0..n_bins {
        let values = &bin_values[bin_idx];

        if values.is_empty() {
            bin_means[bin_idx] = f64::NAN;
            bin_stds[bin_idx] = f64::NAN;
        } else if values.len() == 1 {
            // Single gene in bin - Python sets std to NaN
            bin_means[bin_idx] = values[0];
            bin_stds[bin_idx] = f64::NAN;
        } else {
            // Calculate mean
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            bin_means[bin_idx] = mean;

            // Calculate standard deviation
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

            bin_stds[bin_idx] = variance.sqrt();
        }
    }

    Ok((bin_means, bin_stds))
}

/// Normalize dispersions by subtracting expected values and dividing by standard deviation.
///
/// This creates z-scores for dispersion values, allowing identification of genes
/// with unusually high variability relative to genes with similar expression levels.
///
/// ## Parameters
/// * `log_dispersions` - Log-transformed dispersion values
/// * `bin_indices` - Bin assignment for each gene  
/// * `bin_means` - Expected dispersion for each bin
/// * `bin_stds` - Standard deviation of dispersion for each bin
///
/// ## Returns
/// * Normalized dispersion scores (z-scores)
///
/// ## Formula
/// `normalized_dispersion = (observed_dispersion - expected_dispersion) / std_dispersion`
///
/// Higher values indicate genes that are more variable than expected.
fn normalize_dispersions(
    log_dispersions: &[f64],
    bin_indices: &[usize],
    bin_means: &[f64],
    bin_stds: &[f64],
) -> anyhow::Result<Vec<f64>> {
    let mut normalized_dispersions = vec![0.0; log_dispersions.len()];

    for (i, &disp) in log_dispersions.iter().enumerate() {
        let bin_idx = bin_indices[i];

        if bin_idx >= bin_means.len() {
            normalized_dispersions[i] = f64::NAN;
            continue;
        }

        let mean = bin_means[bin_idx];
        let std = bin_stds[bin_idx];

        if disp.is_nan() || mean.is_nan() || std.is_nan() || std == 0.0 {
            normalized_dispersions[i] = f64::NAN;
        } else {
            normalized_dispersions[i] = (disp - mean) / std;
        }
    }

    Ok(normalized_dispersions)
}

/// Select highly variable genes based on dispersion scores and expression filters.
///
/// Final step of HVG detection that applies thresholds to identify the most
/// biologically relevant genes. Supports both top-N selection and threshold-based selection.
///
/// ## Parameters
/// * `log_means` - Log-transformed mean expression values
/// * `dispersion_norm` - Normalized dispersion scores
/// * `n_top_genes` - Optional: select this many top genes by dispersion
/// * `min_mean` - Minimum mean expression threshold
/// * `max_mean` - Maximum mean expression threshold  
/// * `min_dispersion` - Minimum normalized dispersion threshold
///
/// ## Returns
/// * Boolean mask indicating highly variable genes
///
/// ## Selection Logic
/// - If `n_top_genes` specified: Select top N genes by normalized dispersion
/// - Otherwise: Apply all threshold filters (mean and dispersion)
/// - Genes with NaN dispersions are handled appropriately for each mode
fn subset_genes(
    log_means: &[f64], // These are already log-transformed
    dispersion_norm: &[f64],
    n_top_genes: Option<usize>,
    min_mean: f64,
    max_mean: f64,
    min_dispersion: f64,
) -> anyhow::Result<Vec<bool>> {
    let mut highly_variable = vec![false; log_means.len()];

    if let Some(n_top) = n_top_genes {
        // Python's approach for n_top_genes:
        // 1. First, remove NaN values to compute threshold
        let non_nan_dispersions: Vec<f64> = dispersion_norm
            .iter()
            .filter(|&&d| !d.is_nan())
            .copied()
            .collect();

        if non_nan_dispersions.is_empty() {
            return Ok(highly_variable);
        }

        // Find the nth highest value
        let n_to_select = n_top.min(non_nan_dispersions.len());
        let mut sorted_dispersions = non_nan_dispersions.clone();
        sorted_dispersions.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let threshold = sorted_dispersions[n_to_select - 1];

        // 2. Now apply threshold to nan_to_num version (NaN → -inf)
        for i in 0..dispersion_norm.len() {
            let disp_value = if dispersion_norm[i].is_nan() {
                f64::NEG_INFINITY // Python uses -inf for NaN in final selection
            } else {
                dispersion_norm[i]
            };

            if disp_value >= threshold {
                highly_variable[i] = true;
            }
        }
    } else {
        // Original cutoff-based selection
        // Python applies nan_to_num (NaN → 0) before checking bounds
        let clean_dispersions: Vec<f64> = dispersion_norm
            .iter()
            .map(|&d| if d.is_nan() { 0.0 } else { d })
            .collect();

        // Apply mean filters
        let valid_by_mean: Vec<bool> = log_means
            .iter()
            .map(|&log_mean| log_mean > min_mean && log_mean < max_mean)
            .collect();

        for i in 0..log_means.len() {
            highly_variable[i] = valid_by_mean[i] && clean_dispersions[i] > min_dispersion;
        }
    }

    Ok(highly_variable)
}

/// Compute highly variable genes using the Seurat method.
///
/// The Seurat method models the mean-variance relationship by:
/// 1. Calculating dispersion (variance/mean) for each gene
/// 2. Binning genes by expression level  
/// 3. Computing expected dispersion within each bin
/// 4. Normalizing dispersions to z-scores
/// 5. Selecting genes with high normalized dispersion
///
/// ## Method Details
/// - Uses log1p transformation of means for binning
/// - Handles zero dispersions by setting to NaN
/// - Applies equal-width binning strategy
/// - Stores log-transformed values for consistency with scanpy
///
/// ## Parameters
/// * `adata` - AnnData object with expression data
/// * `x` - Expression matrix
/// * `params` - HVG parameters including thresholds and bin count
///
/// ## Output Columns
/// - `means`: Log1p mean expression
/// - `dispersions`: Log dispersion values
/// - `dispersions_norm`: Z-score normalized dispersions
/// - `highly_variable`: Boolean selection mask
fn compute_seurat_hvg(
    adata: &IMAnnData,
    x: &IMArrayElement,
    params: HVGParams,
) -> anyhow::Result<()> {
    let n_obs = adata.n_obs();

    // Calculate means from raw counts
    let raw_means: Vec<f64> = x
        .sum_whole(&Direction::COLUMN)?
        .iter()
        .map(|sum: &f64| sum / n_obs as f64)
        .collect();

    let variances: Vec<f64> = x.variance_whole::<u32, f64>(&Direction::COLUMN)?;

    // Calculate dispersions with proper handling of zero means
    let dispersions: Vec<f64> = raw_means
        .iter()
        .zip(variances.iter())
        .map(|(&mean, &var)| {
            let safe_mean = if mean > 1e-12 { mean } else { 1e-12 };
            var / safe_mean
        })
        .collect();

    // For Seurat flavor, use log1p of means for binning and storage
    // This matches what Python does AFTER reverting log normalization
    let log1p_means: Vec<f64> = raw_means.iter().map(|&x| (x + 1.0).ln()).collect();

    // Log dispersions with NaN for zero dispersions (matching Python)
    let log_dispersions: Vec<f64> = dispersions
        .iter()
        .map(|&x| {
            if x > 0.0 {
                x.ln()
            } else {
                f64::NAN // Python sets dispersion[dispersion == 0] = np.nan
            }
        })
        .collect();

    let n_bins = params.n_bins;

    // Use equal-width binning on log1p_means (like Python's pd.cut)
    let (bin_indices, _) = equal_width_binning(&log1p_means, n_bins)?;

    // Calculate mean and std for each bin
    let (mut bin_means, mut bin_stds) =
        calculate_bin_stats(&log_dispersions, &bin_indices, n_bins)?;

    // Handle single-gene bins (like Python's _postprocess_dispersions_seurat)
    postprocess_seurat_dispersions(&mut bin_means, &mut bin_stds)?;

    // Normalize dispersions
    let normalized_dispersions =
        normalize_dispersions(&log_dispersions, &bin_indices, &bin_means, &bin_stds)?;

    // Select highly variable genes using raw means for filtering
    let highly_variable = subset_genes(
        &log1p_means, // Pass log-transformed means
        &normalized_dispersions,
        params.n_top_genes,
        params.min_mean,
        params.max_mean,
        params.min_dispersion,
    )?;

    // Store results - IMPORTANT: Store log1p means to match Python
    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("means".into(), log1p_means))?; // Store log1p means like Python
    var_df.with_column(Column::new("dispersions".into(), log_dispersions))?; // Store log dispersions
    var_df.with_column(Column::new(
        "dispersions_norm".into(),
        normalized_dispersions,
    ))?;
    var_df.with_column(Column::new("highly_variable".into(), highly_variable))?;

    adata.var().set_data(var_df)
}

/// Placeholder for Cell Ranger highly variable gene detection.
///
/// The Cell Ranger method follows a similar approach to Seurat but with
/// modifications specific to the 10x Genomics processing pipeline.
/// This implementation is planned for future development.
///
/// ## TODO
/// - Implement Cell Ranger-specific normalization
/// - Add proper mean-variance modeling for 10x data
/// - Include Cell Ranger-specific parameter handling
fn compute_cell_ranger_hvg(
    _adata: &IMAnnData,
    _x: &IMArrayElement,
    _params: HVGParams,
) -> anyhow::Result<()> {
    todo!("Cell Ranger flavor is not implemented yet!")
}

/// Compute highly variable genes using Support Vector Regression (SVR).
///
/// The SVR method uses machine learning to model the mean-variance relationship:
/// 1. Computes log(mean) and log(variance) for each gene
/// 2. Fits SVR model to predict variance from mean
/// 3. Calculates residuals (observed - predicted variance)
/// 4. Standardizes residuals to identify outliers
/// 5. Selects genes with high standardized residuals
///
/// ## Advantages
/// - More flexible modeling of mean-variance relationship
/// - Can capture non-linear trends
/// - Less sensitive to binning artifacts
///
/// ## Disadvantages  
/// - More computationally intensive
/// - Requires tuning of SVR hyperparameters
/// - May be less stable with small datasets
///
/// ## Parameters
/// * `adata` - AnnData object with expression data
/// * `x` - Expression matrix
/// * `params` - HVG parameters including selection criteria
///
/// ## Output Columns
/// - `means`: Mean expression per gene
/// - `variances`: Variance per gene
/// - `residuals`: SVR residuals (observed - predicted variance)
/// - `residuals_standardized`: Z-score standardized residuals  
/// - `mean_variance_trend`: Predicted variance from SVR model
/// - `highly_variable`: Boolean selection mask
fn compute_svr_hvg(adata: &IMAnnData, x: &IMArrayElement, params: HVGParams) -> anyhow::Result<()> {
    let n_obs = adata.n_obs();
    let means: Vec<f64> = x
        .sum_whole(&Direction::COLUMN)?
        .iter()
        .map(|sum: &f64| sum / n_obs as f64)
        .collect();

    let variances: Vec<f64> = x.variance_whole::<u32, f64>(&Direction::COLUMN)?;

    let log_means: Vec<f64> = means.iter().map(|&x| x.ln()).collect();
    let log_variances: Vec<f64> = variances.iter().map(|&x| x.ln()).collect();

    let (residuals, y_pred) = fit_svr(&log_means, &log_variances)?;
    let standardized_results = standardize_log_form_vec(&residuals);

    let mut highly_variable = vec![false; means.len()];
    if let Some(n_top) = params.n_top_genes {
        let mut indices: Vec<usize> = (0..standardized_results.len()).collect();
        indices.sort_by(|&a, &b| {
            standardized_results[b]
                .partial_cmp(&standardized_results[a])
                .unwrap()
        });
        for &idx in indices.iter().take(n_top) {
            if means[idx] >= params.min_mean && means[idx] <= params.max_mean {
                highly_variable[idx] = true;
            }
        }
    } else {
        for i in 0..means.len() {
            highly_variable[i] = means[i] >= params.min_mean
                && means[i] <= params.max_mean
                && standardized_results[i] > params.min_dispersion;
        }
    }

    let mut var_df = adata.var().get_data();
    var_df.with_column(Column::new("means".into(), means))?;
    var_df.with_column(Column::new("variances".into(), variances))?;
    var_df.with_column(Column::new("residuals".into(), residuals))?;
    var_df.with_column(Column::new("highly_variable".into(), highly_variable))?;
    var_df.with_column(Column::new(
        "residuals_standardized".into(),
        standardized_results,
    ))?;
    var_df.with_column(Column::new("mean_variance_trend".into(), y_pred))?;

    adata.var().set_data(var_df)
}
