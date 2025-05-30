use ndarray::parallel::prelude::IndexedParallelIterator;
use crate::memory::utils::{create_dataframe_from_map, create_string_dataframe_from_map};
use anndata::data::{DynCsrMatrix, DynScalar};
use anndata::{ArrayData, Data};
use anndata_memory::{IMAnnData, IMElement};
use nalgebra_sparse::CsrMatrix;
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::ParallelIterator;
use num_traits::{Float, FromPrimitive, NumCast};
use polars::datatypes::CategoricalOrdering;
use polars::datatypes::DataType;
use polars::prelude::LogicalType;
use single_statistics::testing::correction::{
    benjamini_hochberg_correction, benjamini_yekutieli_correction, bonferroni_correction,
    hochberg_correction, holm_bonferroni_correction, storey_qvalues,
};
use single_statistics::testing::inference::nonparametric::mann_whitney;
use single_statistics::testing::inference::parametric::t_test;
use single_statistics::testing::inference::MatrixStatTests;
use single_statistics::testing::{Alternative, TTestType, TestMethod, TestResult};
use single_utilities::traits::{FloatOps, FloatOpsTS};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Deref;

#[derive(Clone)]
pub enum CorrectionMethod {
    Bonferroni,
    BejaminiHochberg,
    BenjaminiYekutieli,
    HolmBonferroni,
    Hochberg,
    StoreyQValue,
}

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
            _ => todo!(),
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

struct DifferentialExpressionResults {
    scores: HashMap<String, Vec<f64>>,
    pvals: HashMap<String, Vec<f64>>,
    pvals_adj: HashMap<String, Vec<f64>>,
    logfoldchanges: HashMap<String, Vec<f64>>,
    gene_names: HashMap<String, Vec<String>>,
}

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
                    return Err(anyhow::anyhow!("No cells fround in reference group: rest"));
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

        scores_map.insert(
            group.clone(),
            group_results
                .scores
                .into_iter()
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect(),
        );
        pvals_map.insert(
            group.clone(),
            group_results
                .pvals
                .into_iter()
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect(),
        );
        pvals_adj_map.insert(
            group.clone(),
            group_results
                .pvals_adj
                .into_iter()
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect(),
        );
        logfoldchanges_map.insert(
            group.clone(),
            group_results
                .logfoldchanges
                .into_iter()
                .map(|x| x.to_f64().unwrap_or(0.0))
                .collect(),
        );
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

struct GroupTestResults<T>
where
    T: FloatOps,
{
    scores: Vec<T>,
    pvals: Vec<T>,
    pvals_adj: Vec<T>,
    logfoldchanges: Vec<T>,
    gene_names: Vec<String>,
}

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
) -> anyhow::Result<GroupTestResults<T>>
where
    T: FloatOpsTS,
    CsrMatrix<T>: MatrixStatTests<T>,
{
    let n_cols = csr_matrix.ncols();
    let n_rows = csr_matrix.nrows();
    let pseudocount = T::from(pseudocount).unwrap();
    
    let group_size = T::from(group_indices.len()).unwrap();
    let ref_size = T::from(reference_indices.len()).unwrap();
    let group_inv = T::one() / group_size;
    let ref_inv = T::one() / ref_size;
    
    let group_set: std::collections::HashSet<usize> = group_indices.iter().copied().collect();
    let ref_set: std::collections::HashSet<usize> = reference_indices.iter().copied().collect();
    
    let mut scores = Vec::with_capacity(n_cols);
    let mut pvals = Vec::with_capacity(n_cols);
    let mut logfoldchanges = Vec::with_capacity(n_cols);
    
    let mut group_sums = vec![T::zero(); n_cols];
    let mut ref_sums = vec![T::zero(); n_cols];
    let mut group_values_per_col: Vec<Vec<T>> =
        vec![Vec::with_capacity(group_indices.len()); n_cols];
    let mut ref_values_per_col: Vec<Vec<T>> =
        vec![Vec::with_capacity(reference_indices.len()); n_cols];
    
    for row in 0..n_rows {
        let is_group = group_set.contains(&row);
        let is_ref = ref_set.contains(&row);

        if !is_group && !is_ref {
            continue;
        }
        
        if let row_data = csr_matrix.row(row) {
            for (&col, &value) in row_data.col_indices().iter().zip(row_data.values()) {
                if is_group {
                    group_sums[col] += value;
                    group_values_per_col[col].push(value);
                }
                if is_ref {
                    ref_sums[col] += value;
                    ref_values_per_col[col].push(value);
                }
            }
        }
    }
    
    const CHUNK_SIZE: usize = 64; 

    let chunk_results: Vec<Vec<(T, T, T)>> = (0..n_cols)
        .into_par_iter()
        .chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut chunk_scores = Vec::with_capacity(chunk.len());
            let mut chunk_pvals = Vec::with_capacity(chunk.len());
            let mut chunk_lfcs = Vec::with_capacity(chunk.len());

            for col in chunk {
                let group_values = &group_values_per_col[col];
                let ref_values = &ref_values_per_col[col];
                
                let mut padded_group_values = group_values.clone();
                let mut padded_ref_values = ref_values.clone();

                padded_group_values.resize(group_indices.len(), T::zero());
                padded_ref_values.resize(reference_indices.len(), T::zero());
                
                let test_result = match method {
                    TestMethod::TTest(test_type) => t_test(
                        &padded_group_values,
                        &padded_ref_values,
                        test_type,
                        Alternative::TwoSided,
                    ),
                    TestMethod::MannWhitney => mann_whitney(
                        &padded_group_values,
                        &padded_ref_values,
                        Alternative::TwoSided,
                    ),
                    _ => TestResult::new(T::zero(), T::one()),
                };
                
                let log_fc = if compute_lfc {
                    let mean_group = group_sums[col] * group_inv + pseudocount;
                    let mean_ref = ref_sums[col] * ref_inv + pseudocount;
                    (mean_group / mean_ref).log2()
                } else {
                    T::zero()
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
    
    gene_indices.sort_unstable_by(|&a, &b| {
        match pvals_adj[a].partial_cmp(&pvals_adj[b]) {
            Some(Ordering::Equal) => {
                pvals[a].partial_cmp(&pvals[b]).unwrap_or(Ordering::Equal)
            }
            Some(ord) => ord,
            None => Ordering::Equal,
        }
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

fn get_unique_groups(adata: &IMAnnData, groupby: &str) -> anyhow::Result<Vec<String>> {
    let group_col = adata.obs().get_column_from_df(groupby)?;
    let mut all_groups = Vec::new();

    match group_col.dtype() {
        DataType::String => {
            let string_col = group_col.str()?;
            let mut unique_groups = std::collections::HashSet::new();

            for i in 0..string_col.len() {
                if let Some(value) = string_col.get(i) {
                    unique_groups.insert(value.to_string());
                }
            }

            all_groups = unique_groups.into_iter().collect();
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

            all_groups = unique_groups.into_iter().collect();
        }
        DataType::Categorical(Some(mapping), ordering) => {
            let categories = mapping.get_categories();
            let mut unique_groups = Vec::new();

            for i in 0..categories.len() {
                let category = categories.value(i);
                unique_groups.push(category.to_string());
            }

            match ordering {
                CategoricalOrdering::Physical => {
                    // nothing to do here
                }
                CategoricalOrdering::Lexical => {
                    unique_groups.sort();
                }
            }

            all_groups = unique_groups;
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

            all_groups = unique_groups.into_iter().collect();
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported data type for groupby column: {:?}",
                other
            ));
        }
    }

    if !matches!(group_col.dtype(), DataType::Categorical(Some(_), _)) {
        all_groups.sort();
    }
    Ok(all_groups)
}

fn filter_groups_to_test(
    all_groups: &[String],
    groups: Option<&[&str]>,
) -> anyhow::Result<Vec<String>> {
    match groups {
        Some(g) => {
            let mut filtered = Vec::new();
            for &group in g {
                if all_groups.contains(&group.to_string()) {
                    filtered.push(group.to_string());
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

fn resolve_reference_group(
    all_groups: &[String],
    reference: Option<&str>,
) -> anyhow::Result<Option<String>> {
    match reference {
        Some(ref_group) => {
            if ref_group == "rest" {
                Ok(None)
            } else if all_groups.contains(&ref_group.to_string()) {
                Ok(Some(ref_group.to_string()))
            } else {
                Err(anyhow::anyhow!("Reference group '{}' not found", ref_group))
            }
        }
        None => Ok(None),
    }
}

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
fn perform_test<T>(
    matrix: &CsrMatrix<T>,
    group_indices: &[usize],
    reference_indices: &[usize],
    method: TestMethod,
) -> anyhow::Result<Vec<TestResult<T>>>
where
    T: FloatOpsTS,
    CsrMatrix<T>: MatrixStatTests<T>,
{
    let n_cols = matrix.ncols(); 
    
    let results: Vec<TestResult<T>> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            let mut group_values: Vec<T> = Vec::with_capacity(group_indices.len());
            for &row in group_indices {
                if let Some(entry) = matrix.get_entry(row, col) {
                    let value = entry.into_value();
                    group_values.push(value);
                } else {
                    group_values.push(T::zero()); 
                }
            }
            
            let mut reference_values: Vec<T> = Vec::with_capacity(reference_indices.len());
            for &row in reference_indices {
                if let Some(entry) = matrix.get_entry(row, col) {
                    let value = entry.into_value();
                    reference_values.push(value);
                } else {
                    reference_values.push(T::zero());
                }
            }
            
            match method {
                TestMethod::TTest(test_type) => t_test(
                    &group_values,
                    &reference_values,
                    test_type,
                    Alternative::TwoSided,
                ),
                TestMethod::MannWhitney => {
                    mann_whitney(&group_values, &reference_values, Alternative::TwoSided)
                }
                _ => TestResult::new(T::zero(), T::one()),
            }
        })
        .collect();

    Ok(results)
}

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
    println!("Storing groups");
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

    // Helper function to create a test AnnData object with synthetic data
    fn create_test_anndata() -> anyhow::Result<IMAnnData> {
        // Create a synthetic gene expression matrix with clear patterns
        // Matrix dimensions: 10 genes × 12 cells (6 in group A, 6 in group B)
        //
        // Patterns:
        // - Genes 0-2: Highly expressed in group A, low in group B
        // - Genes 3-5: Highly expressed in group B, low in group A
        // - Genes 6-9: No significant difference between groups
        let rows: Vec<usize> = vec![
            // Genes 0-2: High in A, low in B
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, // Genes 3-5: High in B, low in A
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, // Genes 6-9: No difference
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        ];

        let cols: Vec<usize> = vec![
            // Genes 0-2: High in A (cols 0-5), low in B (cols 6-11)
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11,
            // Genes 3-5: Low in A (cols 0-5), high in B (cols 6-11)
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, // Genes 6-9: No difference
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        ];

        let vals: Vec<f32> = vec![
            // Genes 0-2: High in A (cols 0-5), low in B (cols 6-11)
            10.0, 10.2, 9.8, 10.5, 10.3, 9.7, 1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 12.0, 11.8, 12.2, 11.5,
            12.5, 11.7, 1.5, 1.7, 1.3, 1.6, 1.4, 1.8, 11.0, 11.3, 10.7, 11.2, 10.8, 11.4, 1.2, 1.1,
            1.3, 0.9, 1.4, 1.0, // Genes 3-5: Low in A (cols 0-5), high in B (cols 6-11)
            1.5, 1.3, 1.7, 1.4, 1.8, 1.2, 8.0, 8.2, 7.8, 8.5, 7.7, 8.3, 1.8, 1.6, 2.0, 1.5, 1.9,
            1.7, 9.0, 8.8, 9.2, 8.7, 9.3, 8.9, 1.2, 1.4, 1.0, 1.3, 0.9, 1.1, 7.5, 7.7, 7.3, 7.8,
            7.2, 7.9, // Genes 6-9: No difference
            5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 5.1, 4.9, 5.3, 4.7, 5.2, 5.0, 4.7, 4.5, 4.9, 4.6, 5.0,
            4.8, 4.8, 5.0, 4.6, 4.9, 4.7, 5.1, 5.2, 5.0, 5.4, 4.8, 5.3, 5.1, 5.0, 5.2, 4.8, 5.3,
            4.9, 5.1, 3.0, 3.2, 2.8, 3.1, 2.9, 3.3, 3.2, 2.8, 3.4, 2.9, 3.3, 3.1,
        ];

        // Create a CooMatrix first, then convert to CsrMatrix
        let coo = CooMatrix::try_from_triplets(10, 12, rows, cols, vals).unwrap();
        let csr = CsrMatrix::from(&coo);

        // Create observation annotations (cell metadata)
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

        // Create variable annotations (gene metadata)
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

    // Test 1: Basic functionality test with default parameters
    #[test]
    fn test_basic_rank_genes() -> anyhow::Result<()> {
        let adata = create_test_anndata()?;

        // Run rank_gene_groups with default parameters
        rank_gene_groups(
            &adata,
            "group",                            // groupby
            Some("B"),                          // reference
            Some(&["A"]),                       // test only group A
            None,                               // key_added (default)
            None,                               // method (default t-test)
            None,                               // n_genes (default)
            CorrectionMethod::BejaminiHochberg, // correction method
            None,                               // compute_logfoldchanges (default true)
            None,                               // pseudocount (default 1.0)
        )?;

        // Check that results were stored in uns
        let uns = adata.uns();
        let scores = uns.get_data("rank_genes_groups_scores");
        let pvals = uns.get_data("rank_genes_groups_pvals");
        let pvals_adj = uns.get_data("rank_genes_groups_pvals_adj");
        let logfc = uns.get_data("rank_genes_groups_logfoldchanges");
        let names = uns.get_data("rank_genes_groups_names");

        // Verify that results exist (a minimal check)
        assert!(scores.is_ok());
        assert!(pvals.is_ok());
        assert!(pvals_adj.is_ok());
        assert!(logfc.is_ok());
        assert!(names.is_ok());

        Ok(())
    }

    // Test 2: Validate results match expected patterns
    #[test]
    fn test_validate_results() -> anyhow::Result<()> {
        let adata = create_test_anndata()?;

        // Run with few returned genes to simplify validation
        rank_gene_groups(
            &adata,
            "group",
            Some("B"),
            Some(&["A"]),
            Some("test_result"), // with a specific key
            Some(TestMethod::TTest(TTestType::Welch)),
            Some(6), // return top 6 genes
            CorrectionMethod::BejaminiHochberg,
            Some(true), // compute log fold changes
            Some(1.0),  // pseudocount
        )?;

        // Todo add these checks:
        // 1. The top 6 genes should include genes 0-5
        // 2. Genes 0-2 should have positive log fold changes (A > B)
        // 3. Genes 3-5 should have negative log fold changes (A < B)
        // 4. P-values should be very small for genes 0-5
        // 5. The genes should be ranked by adjusted p-value

        // Extract the gene_names for checking
        let uns = adata.uns();
        let names_array = uns.get_data("rank_genes_groups_test_result_logfoldchanges");

        // Check the gene names exist
        assert!(names_array.is_ok());
        let gene_names = names_array?.get_data()?;
        let gene_names = match gene_names {
            Data::ArrayData(array_data) => match array_data {
                ArrayData::DataFrame(df) => {
                    assert_eq!(df.height(), 6)
                }
                other => {
                    panic!("This is not the dataformat expected. It should be an dataframe, found {:?}!", other)
                }
            },
            Data::Scalar(_) => {
                panic!("This is not the data format expected. This should be an dataframe, but found scalar")
            }
            Data::Mapping(_) => {
                panic!("This is not the data format expected. This should be an dataframe, but found mapping")
            }
        };

        Ok(())
    }
}
