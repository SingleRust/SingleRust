use crate::memory::utils::{create_dataframe_from_map, create_string_dataframe_from_map};
use anndata::data::{DynCsrMatrix, DynScalar};
use anndata::{ArrayData, Data};
use anndata_memory::{IMAnnData, IMElement};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, FromPrimitive, NumCast};
use polars::datatypes::DataType;
use single_algebra::statistics::correction::{
    benjamini_hochberg_correction, benjamini_yekutieli_correction, bonferroni_correction,
    hochberg_correction, holm_bonferroni_correction, storey_qvalues,
};
use single_algebra::statistics::effect::calculate_log2_fold_change;
use single_algebra::statistics::inference::MatrixStatTests;
use single_algebra::statistics::{Alternative, TTestType, TestMethod, TestResult};
use smartcore::linalg::basic::arrays::Array;
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
    let n_genes = n_genes.unwrap_or(100);

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
    T: Float + NumCast + FromPrimitive + Send + Sync + Debug,
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

struct GroupTestResults {
    scores: Vec<f64>,
    pvals: Vec<f64>,
    pvals_adj: Vec<f64>,
    logfoldchanges: Vec<f64>,
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
) -> anyhow::Result<GroupTestResults>
where
    T: Float + NumCast + FromPrimitive + Send + Sync + Debug,
    CsrMatrix<T>: MatrixStatTests<T>,
{
    let test_results = perform_test(csr_matrix, group_indices, reference_indices, method)?;

    let scores: Vec<f64> = test_results.iter().map(|r| r.statistic).collect();
    let pvals: Vec<f64> = test_results.iter().map(|r| r.p_value).collect();
    let pvals_adj = apply_correction(&pvals, correction_method)?;

    let logfoldchanges = if compute_lfc {
        let mut lfc_vec = Vec::with_capacity(csr_matrix.nrows());
        for row in 0..csr_matrix.nrows() {
            let lfc = calculate_log2_fold_change(
                csr_matrix,
                row,
                group_indices,
                reference_indices,
                pseudocount,
            )
            .unwrap_or(0.0);
            lfc_vec.push(lfc);
        }
        lfc_vec
    } else {
        vec![0.0; csr_matrix.nrows()]
    };

    let mut gene_indices: Vec<usize> = (0..pvals_adj.len()).collect();
    gene_indices.sort_by(|&a, &b| {
        pvals_adj[a]
            .partial_cmp(&pvals_adj[b])
            .unwrap_or(Ordering::Equal)
    });

    if gene_indices.len() > n_genes {
        gene_indices.truncate(n_genes);
    }

    let mut ordered_scores = Vec::with_capacity(gene_indices.len());
    let mut ordered_pvals = Vec::with_capacity(gene_indices.len());
    let mut ordered_pvals_adj = Vec::with_capacity(gene_indices.len());
    let mut ordered_logfoldchanges = Vec::with_capacity(gene_indices.len());
    let mut ordered_gene_names = Vec::with_capacity(gene_indices.len());

    for &idx in &gene_indices {
        ordered_scores.push(scores[idx]);
        ordered_pvals.push(pvals[idx]);
        ordered_pvals_adj.push(pvals_adj[idx]);
        ordered_logfoldchanges.push(logfoldchanges[idx]);

        let gene_name = var_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("Gene_{}", idx));
        ordered_gene_names.push(gene_name);
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
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported data type for groupby column: {:?}",
                other
            ));
        }
    }

    all_groups.sort();
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
                Ok(None) // Special case for comparing against all other cells
            } else if all_groups.contains(&ref_group.to_string()) {
                Ok(Some(ref_group.to_string()))
            } else {
                Err(anyhow::anyhow!("Reference group '{}' not found", ref_group))
            }
        }
        None => Ok(None), // Default to "rest" comparison
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
    group1_indices: &[usize],
    group2_indices: &[usize],
    method: TestMethod,
) -> anyhow::Result<Vec<TestResult>>
where
    T: Float + NumCast + FromPrimitive + Send + Sync + std::fmt::Debug,
    CsrMatrix<T>: MatrixStatTests<T>,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Group indices cannot be empty: group1 has {}, group2 has: {}",
            group1_indices.len(),
            group2_indices.len()
        ));
    }

    let nrows = matrix.nrows();
    for &idx in group1_indices.iter().chain(group2_indices.iter()) {
        if idx >= nrows {
            return Err(anyhow::anyhow!(
                "Cell index {} is out of bounds (nrows = {})",
                idx,
                nrows
            ));
        }
    }

    match method {
        TestMethod::TTest(ttype) => {
            matrix.t_test(group1_indices, group2_indices, ttype, Alternative::TwoSided)
        }
        TestMethod::MannWhitney => {
            matrix.mann_whitney_test(group1_indices, group2_indices, Alternative::TwoSided)
        }
        _ => Err(anyhow::anyhow!(
            "Test method {:?} hasn't been implemented just yet.",
            method
        )),
    }
}

fn apply_correction(p_value: &[f64], method: CorrectionMethod) -> anyhow::Result<Vec<f64>> {
    match method {
        CorrectionMethod::Bonferroni => bonferroni_correction(p_value),
        CorrectionMethod::BejaminiHochberg => benjamini_hochberg_correction(p_value),
        CorrectionMethod::BenjaminiYekutieli => benjamini_yekutieli_correction(p_value),
        CorrectionMethod::HolmBonferroni => holm_bonferroni_correction(p_value),
        CorrectionMethod::Hochberg => hochberg_correction(p_value),
        CorrectionMethod::StoreyQValue => storey_qvalues(p_value, 0.5),
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
