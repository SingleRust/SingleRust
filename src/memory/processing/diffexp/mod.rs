//! # Differential Expression Analysis
//!
//! This module provides high-performance differential expression analysis for single-cell RNA-seq data.
//! It implements statistical tests to identify genes that are significantly differentially expressed
//! between groups of cells, similar to scanpy's `rank_genes_groups` functionality.
//!
//! ## Features
//!
//! - **Multiple Statistical Tests**: Support for parametric (t-test variants) and non-parametric (Mann-Whitney) tests
//! - **Parallel Processing**: Efficient parallel computation using Rust's rayon for large datasets
//! - **Multiple Testing Correction**: Various correction methods (Bonferroni, Benjamini-Hochberg, etc.)
//! - **Flexible Group Comparisons**: Compare specific groups vs reference groups or vs all other cells
//! - **Effect Size Calculation**: Computation of log fold changes with customizable pseudocounts
//!
//! ## Statistical Methods
//!
//! ### Parametric Tests
//! - **Student's t-test**: Assumes equal variances between groups
//! - **Welch's t-test**: Does not assume equal variances (default, recommended)
//!
//! ### Non-parametric Tests  
//! - **Mann-Whitney U test**: Rank-based test, no distributional assumptions
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use single_rust::memory::processing::diffexp::{rank_gene_groups, CorrectionMethod};
//! use single_statistics::testing::{TestMethod, TTestType};
//!
//! // Basic differential expression: Compare group A vs group B
//! rank_gene_groups(
//!     &adata,
//!     "cell_type",           // Column in adata.obs with group labels
//!     Some("B"),             // Reference group
//!     Some(&["A"]),          // Groups to test
//!     None,                  // Use default key "rank_genes_groups"
//!     None,                  // Use default Welch's t-test
//!     None,                  // Return all genes
//!     CorrectionMethod::BejaminiHochberg,
//!     None,                  // Compute log fold changes
//!     None,                  // Use default pseudocount
//! )?;
//!
//! // Compare each group vs all other cells
//! rank_gene_groups(
//!     &adata,
//!     "cell_type",
//!     Some("rest"),          // Use all other cells as reference
//!     None,                  // Test all groups
//!     Some("vs_rest"),       // Custom key for results
//!     Some(TestMethod::MannWhitney), // Use non-parametric test
//!     Some(50),              // Return top 50 genes per group
//!     CorrectionMethod::Bonferroni,
//!     Some(true),
//!     Some(0.5),             // Custom pseudocount
//! )?;
//! ```
//!
//! ## Output Format
//!
//! Results are stored in `adata.uns` with structured keys:
//! - `{key}_scores`: Test statistics for each gene and group
//! - `{key}_pvals`: Raw p-values
//! - `{key}_pvals_adj`: Multiple testing corrected p-values  
//! - `{key}_logfoldchanges`: Log2 fold changes
//! - `{key}_names`: Gene names ranked by significance
//! - `{key}_params_*`: Analysis parameters for reproducibility
//!
//! ## Performance Notes
//!
//! - Uses parallel processing with optimized chunk sizes for large datasets
//! - Pre-computes summary statistics for t-tests to avoid redundant calculations
//! - Optimized sparse matrix operations for memory efficiency
//! - Results are sorted by adjusted p-values, then raw p-values for ranking

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Deref;

use anndata::data::{DynCsrMatrix, DynScalar};
use anndata::{ArrayData, Data};
use anndata_memory::{IMAnnData, IMElement};
use nalgebra_sparse::CsrMatrix;
use ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use polars::datatypes::{CategoricalOrdering, DataType};
use single_statistics::testing::correction::{
    benjamini_hochberg_correction, benjamini_yekutieli_correction, bonferroni_correction,
    hochberg_correction, holm_bonferroni_correction, storey_qvalues,
};
use single_statistics::testing::inference::nonparametric::mann_whitney_optimized;
use single_statistics::testing::inference::parametric::fast_t_test_from_sums;
use single_statistics::testing::inference::MatrixStatTests;
use single_statistics::testing::{Alternative, TTestType, TestMethod, TestResult};
use single_utilities::traits::{FloatOps, FloatOpsTS};

use crate::memory::utils::{create_dataframe_from_map, create_string_dataframe_from_map};

/// Chunk size for parallel processing of statistical tests
const PARALLEL_CHUNK_SIZE: usize = 64;

/// Multiple testing correction methods for controlling family-wise error rate or false discovery rate.
///
/// When testing thousands of genes simultaneously, the probability of false positives increases
/// dramatically. These correction methods adjust p-values to control for multiple testing.
///
/// ## Methods
///
/// - **Bonferroni**: Conservative method controlling family-wise error rate (FWER)
/// - **BejaminiHochberg**: Controls false discovery rate (FDR), less conservative than Bonferroni
/// - **BenjaminiYekutieli**: More conservative FDR control for dependent tests
/// - **HolmBonferroni**: Step-down method, more powerful than Bonferroni
/// - **Hochberg**: Step-up method, requires independence assumption
/// - **StoreyQValue**: Estimates q-values using Storey's method
///
/// ## Recommendations
///
/// - Use **BejaminiHochberg** for most single-cell differential expression analyses
/// - Use **Bonferroni** when very strict control of false positives is needed
/// - Use **StoreyQValue** when you need q-value estimates rather than adjusted p-values
#[derive(Clone)]
pub enum CorrectionMethod {
    /// Bonferroni correction: p_adj = p * n_tests
    Bonferroni,
    /// Benjamini-Hochberg FDR correction (recommended for most analyses)
    BejaminiHochberg,
    /// Benjamini-Yekutieli FDR correction for dependent tests
    BenjaminiYekutieli,
    /// Holm-Bonferroni step-down method
    HolmBonferroni,
    /// Hochberg step-up method
    Hochberg,
    /// Storey q-value estimation
    StoreyQValue,
}

/// Perform differential expression analysis between groups of cells.
///
/// This function implements the rank genes groups functionality similar to scanpy's `rank_genes_groups`.
/// It identifies genes that are differentially expressed between specified groups and a reference group.
///
/// # Arguments
///
/// * `adata` - The AnnData object containing gene expression data
/// * `groupby` - Column name in `adata.obs` that contains group labels
/// * `reference` - Reference group name, or "rest" to use all other cells, or None for automatic selection
/// * `groups` - Specific groups to test, or None to test all groups
/// * `key_added` - Key prefix for storing results in `adata.uns`, defaults to "rank_genes_groups"
/// * `method` - Statistical test method (t-test or Mann-Whitney), defaults to Welch's t-test
/// * `n_genes` - Maximum number of genes to return per group, defaults to all genes
/// * `correction_method` - Multiple testing correction method
/// * `compute_logfoldchanges` - Whether to compute log fold changes, defaults to true
/// * `pseudocount` - Pseudocount for log fold change calculation, defaults to 1.0
///
/// # Returns
///
/// Results are stored in `adata.uns` with the following keys:
/// - `{key}_scores`: Test statistics
/// - `{key}_pvals`: Raw p-values
/// - `{key}_pvals_adj`: Adjusted p-values
/// - `{key}_logfoldchanges`: Log fold changes (if computed)
/// - `{key}_names`: Gene names ranked by significance
/// - `{key}_params_*`: Parameters used for the analysis
///
/// # Examples
///
/// ```rust,ignore
/// use single_rust::memory::processing::diffexp::{rank_gene_groups, CorrectionMethod};
/// use single_statistics::testing::{TestMethod, TTestType};
///
/// // Basic usage
/// rank_gene_groups(
///     &adata,
///     "cell_type",
///     Some("rest"),
///     None,
///     None,
///     None,
///     None,
///     CorrectionMethod::BejaminiHochberg,
///     None,
///     None,
/// )?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn rank_gene_groups(
    adata: &IMAnnData,
    groupby: &str,
    reference: Option<&str>,
    groups: Option<&[&str]>,
    key_added: Option<&str>,
    method: Option<TestMethod>,
    n_genes: Option<usize>,
    correction_method: CorrectionMethod,
    compute_logfoldchanges: Option<bool>,
    pseudocount: Option<f64>,
) -> anyhow::Result<()> {
    let method = method.unwrap_or(TestMethod::TTest(TTestType::Welch));
    let key = key_added.unwrap_or("").to_string();
    let compute_lfc = compute_logfoldchanges.unwrap_or(true);
    let pseudocount = pseudocount.unwrap_or(1.0);
    let n_genes = n_genes.unwrap_or(adata.n_vars());

    let all_groups = get_unique_groups(adata, groupby)?;
    let groups_to_test = filter_groups_to_test(&all_groups, groups)?;
    let reference_group = resolve_reference_group(&all_groups, reference)?;
    let var_names = adata.var_names();

    let x = adata.x();
    let read_guard = x.0.read_inner();
    let data = read_guard.deref();

    let result_maps = match data {
        ArrayData::CsrMatrix(matrix) => match matrix {
            DynCsrMatrix::F32(csr_matrix) => run_differential_expression(
                adata,
                csr_matrix,
                &groups_to_test,
                &reference_group,
                groupby,
                method,
                correction_method,
                compute_lfc,
                pseudocount,
                n_genes,
                &var_names,
            )?,
            DynCsrMatrix::F64(csr_matrix) => run_differential_expression(
                adata,
                csr_matrix,
                &groups_to_test,
                &reference_group,
                groupby,
                method,
                correction_method,
                compute_lfc,
                pseudocount,
                n_genes,
                &var_names,
            )?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported matrix data type. Only F32 and F64 CSR matrices are supported."
                ));
            }
        },
        other => unimplemented!(
            "This feature is currently not implemented for a matrix of type {:?}",
            other
        ),
    };

    store_results(
        adata,
        &key,
        &groups_to_test,
        result_maps.scores,
        result_maps.pvals,
        result_maps.pvals_adj,
        result_maps.logfoldchanges,
        result_maps.gene_names,
        method,
        groupby,
        reference,
    )?;

    Ok(())
}

/// Container for organizing differential expression results across multiple groups.
///
/// This struct holds the complete results of differential expression analysis,
/// with separate collections for each group tested.
struct DifferentialExpressionResults {
    /// Test statistics (t-statistic, U-statistic, etc.) for each group
    scores: HashMap<String, Vec<f64>>,
    /// Raw p-values before multiple testing correction
    pvals: HashMap<String, Vec<f64>>,
    /// Adjusted p-values after multiple testing correction
    pvals_adj: HashMap<String, Vec<f64>>,
    /// Log2 fold changes (group_mean / reference_mean)
    logfoldchanges: HashMap<String, Vec<f64>>,
    /// Gene names ranked by significance for each group
    gene_names: HashMap<String, Vec<String>>,
}

#[allow(clippy::too_many_arguments)]
fn run_differential_expression<T>(
    adata: &IMAnnData,
    csr_matrix: &CsrMatrix<T>,
    groups_to_test: &[String],
    reference_group: &Option<String>,
    groupby: &str,
    method: TestMethod,
    correction_method: CorrectionMethod,
    compute_lfc: bool,
    pseudocount: f64,
    n_genes: usize,
    var_names: &[String],
) -> anyhow::Result<DifferentialExpressionResults>
where
    T: FloatOpsTS,
    CsrMatrix<T>: MatrixStatTests<T>,
{
    let mut scores_map: HashMap<String, Vec<f64>> = HashMap::new();
    let mut pvals_map: HashMap<String, Vec<f64>> = HashMap::new();
    let mut pvals_adj_map: HashMap<String, Vec<f64>> = HashMap::new();
    let mut logfoldchanges_map: HashMap<String, Vec<f64>> = HashMap::new();
    let mut gene_names_map: HashMap<String, Vec<String>> = HashMap::new();

    for group in groups_to_test {
        let group_indices = get_group_indices(adata, groupby, group)?;
        let reference_indices = match reference_group {
            None => {
                let mut all_indices: Vec<usize> = (0..adata.n_obs()).collect();
                all_indices.retain(|&idx| !group_indices.contains(&idx));

                if all_indices.is_empty() {
                    return Err(anyhow::anyhow!("No cells found in reference group: rest"));
                }
                all_indices
            }
            Some(reference_group) => get_group_indices(adata, groupby, reference_group)?,
        };

        let group_results = run_tests_for_group(
            csr_matrix,
            &group_indices,
            &reference_indices,
            method,
            correction_method.clone(),
            compute_lfc,
            pseudocount,
            n_genes,
            var_names,
        )?;

        scores_map.insert(group.clone(), group_results.scores);
        pvals_map.insert(group.clone(), group_results.pvals);
        pvals_adj_map.insert(group.clone(), group_results.pvals_adj);
        logfoldchanges_map.insert(group.clone(), group_results.logfoldchanges);
        gene_names_map.insert(group.clone(), group_results.gene_names);
    }

    Ok(DifferentialExpressionResults {
        scores: scores_map,
        pvals: pvals_map,
        pvals_adj: pvals_adj_map,
        logfoldchanges: logfoldchanges_map,
        gene_names: gene_names_map,
    })
}

/// Container for differential expression results for a single group comparison.
///
/// Holds the complete statistical results for comparing one group against a reference,
/// including test statistics, p-values, effect sizes, and gene rankings.
struct GroupTestResults {
    /// Test statistics ranked by significance
    scores: Vec<f64>,
    /// Raw p-values ranked by significance  
    pvals: Vec<f64>,
    /// Multiple testing corrected p-values ranked by significance
    pvals_adj: Vec<f64>,
    /// Log2 fold changes ranked by significance
    logfoldchanges: Vec<f64>,
    /// Gene names ranked by significance
    gene_names: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_tests_for_group<T>(
    csr_matrix: &CsrMatrix<T>,
    group_indices: &[usize],
    reference_indices: &[usize],
    method: TestMethod,
    correction_method: CorrectionMethod,
    compute_lfc: bool,
    pseudocount: f64,
    n_genes: usize,
    var_names: &[String],
) -> anyhow::Result<GroupTestResults>
where
    T: FloatOpsTS,
    CsrMatrix<T>: MatrixStatTests<T>,
{
    let n_cols = csr_matrix.ncols();
    let n_rows = csr_matrix.nrows();

    let group_size_f64 = group_indices.len() as f64;
    let ref_size_f64 = reference_indices.len() as f64;

    let mut scores: Vec<f64> = Vec::with_capacity(n_cols);
    let mut pvals: Vec<f64> = Vec::with_capacity(n_cols);
    let mut logfoldchanges: Vec<f64> = Vec::with_capacity(n_cols);

    let mut group_sums_f64 = vec![0.0f64; n_cols];
    let mut ref_sums_f64 = vec![0.0f64; n_cols];
    let mut group_sum_sq_f64 = vec![0.0f64; n_cols];
    let mut ref_sum_sq_f64 = vec![0.0f64; n_cols];

    let mut is_group = vec![false; n_rows];
    let mut is_ref = vec![false; n_rows];

    for &idx in group_indices {
        is_group[idx] = true;
    }
    for &idx in reference_indices {
        is_ref[idx] = true;
    }

    for row in 0..n_rows {
        let row_is_group = is_group[row];
        let row_is_ref = is_ref[row];

        if !row_is_group && !row_is_ref {
            continue;
        }

        let row_data = csr_matrix.row(row);
        for (&col, &value) in row_data.col_indices().iter().zip(row_data.values()) {
            let value_f64 = value.to_f64().unwrap_or(0.0);
            if row_is_group {
                group_sums_f64[col] += value_f64;
                group_sum_sq_f64[col] += value_f64 * value_f64;
            }
            if row_is_ref {
                ref_sums_f64[col] += value_f64;
                ref_sum_sq_f64[col] += value_f64 * value_f64;
            }
        }
    }

    let chunk_results: Vec<Vec<(f64, f64, f64)>> = (0..n_cols)
        .into_par_iter()
        .chunks(PARALLEL_CHUNK_SIZE)
        .map(|chunk| {
            let mut chunk_scores = Vec::with_capacity(chunk.len());
            let mut chunk_pvals = Vec::with_capacity(chunk.len());
            let mut chunk_lfcs = Vec::with_capacity(chunk.len());

            for col in chunk {
                let group_mean = group_sums_f64[col] / group_size_f64;
                let ref_mean = ref_sums_f64[col] / ref_size_f64;

                let test_result = if group_mean == 0.0 && ref_mean == 0.0 {
                    TestResult::new(0.0, 1.0)
                } else {
                    match method {
                        TestMethod::TTest(test_type) => {
                            // Use optimized t-test with pre-computed statistics
                            fast_t_test_from_sums(
                                group_sums_f64[col],
                                group_sum_sq_f64[col],
                                group_size_f64,
                                ref_sums_f64[col],
                                ref_sum_sq_f64[col],
                                ref_size_f64,
                                test_type,
                            )
                        }
                        TestMethod::MannWhitney => {
                            // For Mann-Whitney, we still need individual values
                            let mut group_values_f64 = Vec::with_capacity(group_indices.len());
                            let mut ref_values_f64 = Vec::with_capacity(reference_indices.len());

                            for &row_idx in group_indices {
                                let value = if let Some(entry) = csr_matrix.get_entry(row_idx, col)
                                {
                                    entry.into_value().to_f64().unwrap_or(0.0)
                                } else {
                                    0.0
                                };
                                group_values_f64.push(value);
                            }

                            for &row_idx in reference_indices {
                                let value = if let Some(entry) = csr_matrix.get_entry(row_idx, col)
                                {
                                    entry.into_value().to_f64().unwrap_or(0.0)
                                } else {
                                    0.0
                                };
                                ref_values_f64.push(value);
                            }

                            mann_whitney_optimized(
                                &group_values_f64,
                                &ref_values_f64,
                                Alternative::TwoSided,
                            )
                        }
                        _ => TestResult::new(0.0, 1.0),
                    }
                };

                let log_fc = if compute_lfc {
                    if group_mean == 0.0 && ref_mean == 0.0 {
                        0.0
                    } else {
                        let linear_group_mean = group_mean.exp() - 1.0 + pseudocount;
                        let linear_ref_mean = ref_mean.exp() - 1.0 + pseudocount;
                        (linear_group_mean / linear_ref_mean).log2()
                    }
                } else {
                    0.0
                };

                chunk_scores.push(test_result.statistic);
                chunk_pvals.push(test_result.p_value);
                chunk_lfcs.push(log_fc);
            }

            chunk_scores
                .into_iter()
                .zip(chunk_pvals)
                .zip(chunk_lfcs)
                .map(|((s, p), l)| (s, p, l))
                .collect()
        })
        .collect();

    for chunk in chunk_results {
        for (score, pval, lfc) in chunk {
            scores.push(score);
            pvals.push(pval);
            logfoldchanges.push(lfc);
        }
    }

    let pvals_adj = apply_correction(&pvals, correction_method)?;

    let mut gene_indices: Vec<usize> = (0..pvals_adj.len()).collect();

    gene_indices.sort_unstable_by(|&a, &b| match pvals_adj[a].partial_cmp(&pvals_adj[b]) {
        Some(Ordering::Equal) => pvals[a].partial_cmp(&pvals[b]).unwrap_or(Ordering::Equal),
        Some(ord) => ord,
        None => Ordering::Equal,
    });

    gene_indices.truncate(n_genes.min(gene_indices.len()));

    let result_len = gene_indices.len();
    let mut ordered_scores = Vec::with_capacity(result_len);
    let mut ordered_pvals = Vec::with_capacity(result_len);
    let mut ordered_pvals_adj = Vec::with_capacity(result_len);
    let mut ordered_logfoldchanges = Vec::with_capacity(result_len);
    let mut ordered_gene_names = Vec::with_capacity(result_len);

    for &idx in &gene_indices {
        unsafe {
            ordered_scores.push(*scores.get_unchecked(idx));
            ordered_pvals.push(*pvals.get_unchecked(idx));
            ordered_pvals_adj.push(*pvals_adj.get_unchecked(idx));
            ordered_logfoldchanges.push(*logfoldchanges.get_unchecked(idx));
            ordered_gene_names.push(var_names.get_unchecked(idx).clone());
        }
    }

    Ok(GroupTestResults {
        scores: ordered_scores,
        pvals: ordered_pvals,
        pvals_adj: ordered_pvals_adj,
        logfoldchanges: ordered_logfoldchanges,
        gene_names: ordered_gene_names,
    })
}

/// Extract unique group labels from a categorical or string column in the observation metadata.
///
/// This function handles different data types commonly used for group labels:
/// - String columns
/// - Integer columns (converted to strings)  
/// - Categorical columns (with proper ordering preservation)
///
/// # Arguments
/// * `adata` - The AnnData object
/// * `groupby` - Column name in `adata.obs` containing group labels
///
/// # Returns
/// Sorted vector of unique group names as strings
fn get_unique_groups(adata: &IMAnnData, groupby: &str) -> anyhow::Result<Vec<String>> {
    let group_col = adata.obs().get_column_from_df(groupby)?;

    let mut all_groups = match group_col.dtype() {
        DataType::String => {
            let string_col = group_col.str()?;
            let mut unique_groups = std::collections::HashSet::new();

            for i in 0..string_col.len() {
                if let Some(value) = string_col.get(i) {
                    unique_groups.insert(value.to_string());
                }
            }

            unique_groups.into_iter().collect()
        }
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let int_col = group_col.i64()?;
            let mut unique_groups = std::collections::HashSet::new();

            for i in 0..int_col.len() {
                if let Some(value) = int_col.get(i) {
                    unique_groups.insert(value.to_string());
                }
            }

            unique_groups.into_iter().collect()
        }
        DataType::Categorical(Some(mapping), ordering) => {
            let categories = mapping.get_categories();
            let mut unique_groups = Vec::new();

            for i in 0..categories.len() {
                let category = categories.value(i);
                unique_groups.push(category.to_string());
            }

            match ordering {
                CategoricalOrdering::Physical => {}
                CategoricalOrdering::Lexical => {
                    unique_groups.sort();
                }
            }

            unique_groups
        }
        DataType::Categorical(None, _) => {
            let string_col = group_col.cast(&DataType::String)?;
            let string_col = string_col.str()?;
            let mut unique_groups = std::collections::HashSet::new();

            for i in 0..string_col.len() {
                if let Some(value) = string_col.get(i) {
                    unique_groups.insert(value.to_string());
                }
            }

            unique_groups.into_iter().collect()
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported data type for groupby column: {:?}",
                other
            ));
        }
    };

    if !matches!(group_col.dtype(), DataType::Categorical(Some(_), _)) {
        all_groups.sort();
    }
    Ok(all_groups)
}

/// Filter and validate the list of groups to test from user input.
///
/// If specific groups are provided, validates they exist in the data.
/// If no groups specified, returns all available groups.
///
/// # Arguments
/// * `all_groups` - Complete list of groups available in the data
/// * `groups` - Optional user-specified subset of groups to test
///
/// # Returns
/// Validated vector of group names to test
fn filter_groups_to_test(
    all_groups: &[String],
    groups: Option<&[&str]>,
) -> anyhow::Result<Vec<String>> {
    match groups {
        Some(g) => {
            let mut filtered = Vec::new();
            for &group in g {
                let group_string = group.to_string();
                if all_groups.contains(&group_string) {
                    filtered.push(group_string);
                } else {
                    return Err(anyhow::anyhow!("Group '{}' not found in data", group));
                }
            }

            if filtered.is_empty() {
                return Err(anyhow::anyhow!("No valid groups to test"));
            }

            Ok(filtered)
        }
        None => Ok(all_groups.to_vec()),
    }
}

/// Resolve the reference group specification to an actual group name or "rest" mode.
///
/// Handles three cases:
/// - Specific group name: validates the group exists
/// - "rest": use all cells not in the test group as reference
/// - None: automatically use "rest" mode
///
/// # Arguments
/// * `all_groups` - Complete list of available groups
/// * `reference` - User-specified reference group or None
///
/// # Returns
/// - `Some(group_name)` for specific reference group
/// - `None` for "rest" mode (use all other cells)
fn resolve_reference_group(
    all_groups: &[String],
    reference: Option<&str>,
) -> anyhow::Result<Option<String>> {
    match reference {
        Some(ref_group) => {
            if ref_group == "rest" {
                Ok(None)
            } else {
                let ref_group_string = ref_group.to_string();
                if all_groups.contains(&ref_group_string) {
                    Ok(Some(ref_group_string))
                } else {
                    Err(anyhow::anyhow!("Reference group '{}' not found", ref_group))
                }
            }
        }
        None => Ok(None),
    }
}

/// Get row indices for cells belonging to a specific group.
///
/// Searches through the groupby column and returns the indices of all cells
/// that match the specified group label. Handles different column data types
/// including strings, integers, and categorical data.
///
/// # Arguments
/// * `adata` - The AnnData object
/// * `groupby` - Column name in `adata.obs` containing group labels
/// * `group` - Group label to search for
///
/// # Returns
/// Vector of 0-based row indices for cells in the specified group
///
/// # Errors
/// Returns error if:
/// - The groupby column doesn't exist
/// - No cells found for the specified group
/// - Data type conversion fails
fn get_group_indices(adata: &IMAnnData, groupby: &str, group: &str) -> anyhow::Result<Vec<usize>> {
    let group_col = adata.obs().get_column_from_df(groupby)?;

    let indices = match group_col.dtype() {
        DataType::String => {
            let string_col = group_col.str()?;
            let mut indices = Vec::new();

            for i in 0..string_col.len() {
                if let Some(value) = string_col.get(i) {
                    if value == group {
                        indices.push(i);
                    }
                }
            }
            indices
        }
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let target = group.parse::<i64>().map_err(|_| {
                anyhow::anyhow!(
                    "Failed to parse group '{}' as integer for numeric column",
                    group
                )
            })?;

            let mut indices = Vec::new();
            let int_col = group_col.i64()?;
            for i in 0..int_col.len() {
                if let Some(value) = int_col.get(i) {
                    if value == target {
                        indices.push(i);
                    }
                }
            }
            indices
        }
        DataType::Categorical(_, _) => {
            let string_col = group_col.cast(&DataType::String)?;
            let string_col = string_col.str()?;
            let mut indices = Vec::new();

            for i in 0..string_col.len() {
                if let Some(value) = string_col.get(i) {
                    if value == group {
                        indices.push(i);
                    }
                }
            }

            indices
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported data type for groupby column: {:?}. Expected String or Integer type.",
                other
            ))
        }
    };

    if indices.is_empty() {
        return Err(anyhow::anyhow!("No cells found for group '{}'", group));
    }

    Ok(indices)
}

/// Apply multiple testing correction to a vector of p-values.
///
/// Converts p-values to the appropriate numeric type and applies the specified
/// correction method. All correction methods are implemented in the `single_statistics`
/// crate with well-tested algorithms.
///
/// # Arguments
/// * `p_value` - Raw p-values to correct
/// * `method` - Multiple testing correction method to apply
///
/// # Returns
/// Vector of corrected p-values in the same order as input
///
/// # Multiple Testing Methods
/// - **Bonferroni**: Most conservative, controls FWER
/// - **Benjamini-Hochberg**: Controls FDR, widely used
/// - **Benjamini-Yekutieli**: More conservative FDR for dependent tests
/// - **Holm-Bonferroni**: Stepwise method, less conservative than Bonferroni
/// - **Hochberg**: Requires independence assumption
/// - **Storey Q-value**: Provides q-value estimates
fn apply_correction<T>(p_value: &[T], method: CorrectionMethod) -> anyhow::Result<Vec<T>>
where
    T: FloatOps,
{
    match method {
        CorrectionMethod::Bonferroni => bonferroni_correction(p_value),
        CorrectionMethod::BejaminiHochberg => benjamini_hochberg_correction(p_value),
        CorrectionMethod::BenjaminiYekutieli => benjamini_yekutieli_correction(p_value),
        CorrectionMethod::HolmBonferroni => holm_bonferroni_correction(p_value),
        CorrectionMethod::Hochberg => hochberg_correction(p_value),
        CorrectionMethod::StoreyQValue => storey_qvalues(p_value, T::from(0.5).unwrap()),
    }
}

#[allow(clippy::too_many_arguments)]
fn store_results(
    adata: &IMAnnData,
    key: &str,
    groups: &[String],
    scores: HashMap<String, Vec<f64>>,
    pvals: HashMap<String, Vec<f64>>,
    pvals_adj: HashMap<String, Vec<f64>>,
    logfoldchanges: HashMap<String, Vec<f64>>,
    gene_names: HashMap<String, Vec<String>>,
    method: TestMethod,
    groupby: &str,
    reference: Option<&str>,
) -> anyhow::Result<()> {
    let scores_df = create_dataframe_from_map(&scores)?;
    let pvals_df = create_dataframe_from_map(&pvals)?;
    let pvals_adj_df = create_dataframe_from_map(&pvals_adj)?;
    let logfoldchanges_df = create_dataframe_from_map(&logfoldchanges)?;
    let gene_names_df = create_string_dataframe_from_map(&gene_names)?;

    let uns = adata.uns();

    let result_key = if key.is_empty() || key == "rank_genes_groups" {
        "rank_genes_groups".to_string()
    } else {
        format!("rank_genes_groups_{}", key)
    };

    uns.add_data(
        format!("{}_scores", result_key),
        IMElement::new(Data::ArrayData(ArrayData::DataFrame(scores_df))),
    )?;

    uns.add_data(
        format!("{}_pvals", result_key),
        IMElement::new(Data::ArrayData(ArrayData::DataFrame(pvals_df))),
    )?;

    uns.add_data(
        format!("{}_pvals_adj", result_key),
        IMElement::new(Data::ArrayData(ArrayData::DataFrame(pvals_adj_df))),
    )?;

    uns.add_data(
        format!("{}_logfoldchanges", result_key),
        IMElement::new(Data::ArrayData(ArrayData::DataFrame(logfoldchanges_df))),
    )?;

    uns.add_data(
        format!("{}_names", result_key),
        IMElement::new(Data::ArrayData(ArrayData::DataFrame(gene_names_df))),
    )?;

    uns.add_data(
        format!("{}_params_reference", result_key),
        IMElement::new(Data::Scalar(DynScalar::String(
            reference.unwrap_or("rest").to_string(),
        ))),
    )?;

    uns.add_data(
        format!("{}_params_method", result_key),
        IMElement::new(Data::Scalar(DynScalar::String(format!("{:?}", method)))),
    )?;

    uns.add_data(
        format!("{}_params_groupby", result_key),
        IMElement::new(Data::Scalar(DynScalar::String(groupby.to_string()))),
    )?;

    let groups_data: Vec<String> = groups.to_vec();
    let groups_array = ndarray::Array1::from_vec(groups_data);
    let groups_dyn = groups_array.into_dyn();

    uns.add_data(
        format!("{}_groups", result_key),
        IMElement::new(Data::from(groups_dyn)),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata_memory::IMAnnData;
    use nalgebra_sparse::{CooMatrix, CsrMatrix};
    use polars::prelude::{DataFrame, NamedFrom, Series};

    fn create_test_anndata() -> anyhow::Result<IMAnnData> {
        let rows: Vec<usize> = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9,
        ];

        let cols: Vec<usize> = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        ];

        let vals: Vec<f32> = vec![
            10.0, 10.2, 9.8, 10.5, 10.3, 9.7, 1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 12.0, 11.8, 12.2, 11.5,
            12.5, 11.7, 1.5, 1.7, 1.3, 1.6, 1.4, 1.8, 11.0, 11.3, 10.7, 11.2, 10.8, 11.4, 1.2, 1.1,
            1.3, 0.9, 1.4, 1.0, 1.5, 1.3, 1.7, 1.4, 1.8, 1.2, 8.0, 8.2, 7.8, 8.5, 7.7, 8.3, 1.8,
            1.6, 2.0, 1.5, 1.9, 1.7, 9.0, 8.8, 9.2, 8.7, 9.3, 8.9, 1.2, 1.4, 1.0, 1.3, 0.9, 1.1,
            7.5, 7.7, 7.3, 7.8, 7.2, 7.9, 5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 5.1, 4.9, 5.3, 4.7, 5.2,
            5.0, 4.7, 4.5, 4.9, 4.6, 5.0, 4.8, 4.8, 5.0, 4.6, 4.9, 4.7, 5.1, 5.2, 5.0, 5.4, 4.8,
            5.3, 5.1, 5.0, 5.2, 4.8, 5.3, 4.9, 5.1, 3.0, 3.2, 2.8, 3.1, 2.9, 3.3, 3.2, 2.8, 3.4,
            2.9, 3.3, 3.1,
        ];

        let coo = CooMatrix::try_from_triplets(10, 12, rows, cols, vals).unwrap();
        let csr = CsrMatrix::from(&coo);

        let mut obs_df = DataFrame::default();
        let names: Vec<String> = vec![
            "c1".into(),
            "c2".into(),
            "c3".into(),
            "c4".into(),
            "c5".into(),
            "c6".into(),
            "c7".into(),
            "c8".into(),
            "c9".into(),
            "c10".into(),
        ];
        let index_col = Series::new("index".into(), names.clone());
        let group_labels = Series::new(
            "group".into(),
            vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        );
        obs_df.with_column(index_col)?;
        obs_df.with_column(group_labels)?;

        let mut var_df = DataFrame::default();
        let g_names: Vec<String> = vec![
            "gene0".into(),
            "gene1".into(),
            "gene2".into(),
            "gene3".into(),
            "gene4".into(),
            "gene5".into(),
            "gene6".into(),
            "gene7".into(),
            "gene8".into(),
            "gene9".into(),
            "gene10".into(),
            "gene11".into(),
        ];
        let gene_names = Series::new("gene_name".into(), g_names.clone());
        var_df.with_column(gene_names)?;
        let adata = IMAnnData::new_extended(ArrayData::from(csr), names, g_names, obs_df, var_df)?;

        Ok(adata)
    }

    #[test]
    fn test_basic_rank_genes() -> anyhow::Result<()> {
        let adata = create_test_anndata()?;

        rank_gene_groups(
            &adata,
            "group",
            Some("B"),
            Some(&["A"]),
            None,
            None,
            None,
            CorrectionMethod::BejaminiHochberg,
            None,
            None,
        )?;

        let uns = adata.uns();
        let scores = uns.get_data("rank_genes_groups_scores");
        let pvals = uns.get_data("rank_genes_groups_pvals");
        let pvals_adj = uns.get_data("rank_genes_groups_pvals_adj");
        let logfc = uns.get_data("rank_genes_groups_logfoldchanges");
        let names = uns.get_data("rank_genes_groups_names");

        assert!(scores.is_ok());
        assert!(pvals.is_ok());
        assert!(pvals_adj.is_ok());
        assert!(logfc.is_ok());
        assert!(names.is_ok());

        Ok(())
    }

    #[test]
    fn test_validate_results() -> anyhow::Result<()> {
        let adata = create_test_anndata()?;

        rank_gene_groups(
            &adata,
            "group",
            Some("B"),
            Some(&["A"]),
            Some("test_result"),
            Some(TestMethod::TTest(TTestType::Welch)),
            Some(6),
            CorrectionMethod::BejaminiHochberg,
            Some(true),
            Some(1.0),
        )?;

        let uns = adata.uns();
        let names_array = uns.get_data("rank_genes_groups_test_result_logfoldchanges");

        assert!(names_array.is_ok());
        let gene_names_data = names_array?.get_data()?;
        match gene_names_data {
            Data::ArrayData(array_data) => match array_data {
                ArrayData::DataFrame(df) => {
                    assert_eq!(df.height(), 6)
                }
                other => {
                    panic!("Expected DataFrame for logfoldchanges, found {:?}", other)
                }
            },
            Data::Scalar(_) => {
                panic!("Expected DataFrame for logfoldchanges, found scalar")
            }
            Data::Mapping(_) => {
                panic!("Expected DataFrame for logfoldchanges, found mapping")
            }
        };

        Ok(())
    }

    #[test]
    fn test_mann_whitney_method() -> anyhow::Result<()> {
        let adata = create_test_anndata()?;

        rank_gene_groups(
            &adata,
            "group",
            Some("rest"),
            Some(&["A"]),
            Some("mann_whitney_test"),
            Some(TestMethod::MannWhitney),
            Some(5),
            CorrectionMethod::Bonferroni,
            Some(true),
            Some(0.5),
        )?;

        let uns = adata.uns();
        let scores = uns.get_data("rank_genes_groups_mann_whitney_test_scores")?;
        let _pvals = uns.get_data("rank_genes_groups_mann_whitney_test_pvals")?;
        let method_param = uns.get_data("rank_genes_groups_mann_whitney_test_params_method")?;

        let method_data = method_param.get_data()?;
        match method_data {
            Data::Scalar(scalar) => match scalar {
                DynScalar::String(s) => {
                    assert!(s.contains("MannWhitney"));
                }
                _ => panic!("Expected string scalar for method parameter"),
            },
            _ => panic!("Expected scalar data for method parameter"),
        }

        match scores.get_data()? {
            Data::ArrayData(ArrayData::DataFrame(df)) => {
                assert_eq!(df.height(), 5, "Should return exactly 5 genes");
            }
            _ => panic!("Expected DataFrame for scores"),
        }

        Ok(())
    }

    #[test]
    fn test_multiple_groups() -> anyhow::Result<()> {
        let adata = create_test_anndata_three_groups()?;

        rank_gene_groups(
            &adata,
            "group",
            None,
            Some(&["A", "B", "C"]),
            Some("multi_group"),
            Some(TestMethod::TTest(TTestType::Student)),
            Some(3),
            CorrectionMethod::BejaminiHochberg,
            Some(true),
            Some(1.0),
        )?;

        let uns = adata.uns();
        let scores = uns.get_data("rank_genes_groups_multi_group_scores")?;
        let _gene_names = uns.get_data("rank_genes_groups_multi_group_names")?;

        match scores.get_data()? {
            Data::ArrayData(ArrayData::DataFrame(df)) => {
                assert_eq!(df.width(), 3, "Should have results for 3 groups");
                assert_eq!(df.height(), 3, "Should have 3 genes per group");

                let column_names = df.get_column_names();
                assert!(column_names.iter().any(|name| name.as_str() == "A"));
                assert!(column_names.iter().any(|name| name.as_str() == "B"));
                assert!(column_names.iter().any(|name| name.as_str() == "C"));
            }
            _ => panic!("Expected DataFrame for scores"),
        }

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> anyhow::Result<()> {
        let adata = create_test_anndata()?;

        let result = rank_gene_groups(
            &adata,
            "group",
            Some("B"),
            Some(&["INVALID_GROUP"]),
            Some("error_test"),
            None,
            None,
            CorrectionMethod::BejaminiHochberg,
            None,
            None,
        );
        assert!(result.is_err(), "Should fail with invalid group name");

        let result = rank_gene_groups(
            &adata,
            "group",
            Some("INVALID_REF"),
            Some(&["A"]),
            Some("error_test2"),
            None,
            None,
            CorrectionMethod::BejaminiHochberg,
            None,
            None,
        );
        assert!(result.is_err(), "Should fail with invalid reference group");

        rank_gene_groups(
            &adata,
            "group",
            Some("B"),
            Some(&["A"]),
            Some("zero_genes"),
            None,
            Some(0),
            CorrectionMethod::BejaminiHochberg,
            None,
            None,
        )?;

        let uns = adata.uns();
        let scores = uns.get_data("rank_genes_groups_zero_genes_scores")?;
        match scores.get_data()? {
            Data::ArrayData(ArrayData::DataFrame(df)) => {
                assert_eq!(df.height(), 0, "Should return 0 genes when n_genes=0");
            }
            _ => panic!("Expected DataFrame for scores"),
        }

        Ok(())
    }

    fn create_test_anndata_three_groups() -> anyhow::Result<IMAnnData> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for cell in 0..15 {
            for gene in 0..9 {
                rows.push(cell);
                cols.push(gene);

                let value = if gene < 3 && cell < 5 {
                    8.0 + (gene as f32) * 0.5 + (cell as f32) * 0.1
                } else if (3..6).contains(&gene) && (5..10).contains(&cell) {
                    7.0 + (gene as f32) * 0.3 + (cell as f32) * 0.1
                } else if gene >= 6 && cell >= 10 {
                    6.0 + (gene as f32) * 0.4 + (cell as f32) * 0.1
                } else {
                    1.0 + (cell as f32) * 0.05
                };

                vals.push(value);
            }
        }

        let coo = CooMatrix::try_from_triplets(15, 9, rows, cols, vals).unwrap();
        let csr = CsrMatrix::from(&coo);

        let mut obs_df = DataFrame::default();
        let cell_names: Vec<String> = (0..15).map(|i| format!("cell_{}", i)).collect();
        let index_col = Series::new("index".into(), cell_names.clone());

        let group_labels = vec![
            "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C",
        ];
        let group_col = Series::new("group".into(), group_labels);

        obs_df.with_column(index_col)?;
        obs_df.with_column(group_col)?;

        let mut var_df = DataFrame::default();
        let gene_names: Vec<String> = (0..9).map(|i| format!("gene_{}", i)).collect();
        let gene_names_col = Series::new("gene_name".into(), gene_names.clone());
        var_df.with_column(gene_names_col)?;

        let adata =
            IMAnnData::new_extended(ArrayData::from(csr), cell_names, gene_names, obs_df, var_df)?;

        Ok(adata)
    }
}
