use std::path::Path;

use anndata::{AnnData, AnnDataOp, Backend};
use anndata_hdf5::H5;
use ndarray::Array1;

use crate::{io::FileScope, utils::ComputationMode};

pub enum FilterLimit {
    Absolute(u32),
    Relative(f64),
    None,
}

pub enum NormalizationMethod {
    TotalCount,
    RelativeCount,
}

impl FilterLimit {
    pub fn is_absolute(&self) -> bool {
        match *self {
            FilterLimit::Absolute(_) => true,
            FilterLimit::Relative(_) => false,
            FilterLimit::None => false,
        }
    }

    pub fn is_relative(&self) -> bool {
        match *self {
            FilterLimit::Absolute(_) => false,
            FilterLimit::Relative(_) => true,
            FilterLimit::None => false,
        }
    }

    pub fn is_none(&self) -> bool {
        match *self {
            FilterLimit::Absolute(_) => false,
            FilterLimit::Relative(_) => false,
            FilterLimit::None => true,
        }
    }
}

impl Clone for FilterLimit {
    fn clone(&self) -> Self {
        match self {
            Self::Absolute(arg0) => Self::Absolute(*arg0),
            Self::Relative(arg0) => Self::Relative(*arg0),
            Self::None => Self::None,
        }
    }
}
// TODO, optimize that!
fn get_removal_mask(
    num_vec: Option<Vec<u32>>,
    sum_vec: Option<Vec<f64>>,
    lower_lim: FilterLimit,
    upper_lim: FilterLimit,
    len: usize,
) -> anyhow::Result<Vec<bool>> {
    let mut keep_cells = vec![true; len];

    if lower_lim.is_absolute() || upper_lim.is_absolute() {
        if let Some(num_genes) = num_vec {
            for (i, &num) in num_genes.iter().enumerate() {
                match lower_lim {
                    FilterLimit::Absolute(lower) if num < lower => keep_cells[i] = false,
                    _ => {}
                }
                if keep_cells[i] {
                    match upper_lim {
                        FilterLimit::Absolute(upper) if num > upper => keep_cells[i] = false,
                        _ => {}
                    }
                }
            }
        }
    }
    
    if lower_lim.is_relative() || upper_lim.is_relative() {
        if let Some(sum_genes) = sum_vec {
            let total_expression: f64 = sum_genes.iter().sum();
    
            for (i, &sum) in sum_genes.iter().enumerate() {
                if keep_cells[i] {
                    let percentage = sum / total_expression;
    
                    match lower_lim {
                        FilterLimit::Relative(lower) if percentage < lower => keep_cells[i] = false,
                        _ => {}
                    }
                    if keep_cells[i] {
                        match upper_lim {
                            FilterLimit::Relative(upper) if percentage > upper => keep_cells[i] = false,
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    Ok(keep_cells)
}

pub fn remove_cells_inplace<B: Backend>(
    anndata: &AnnData<B>,
    lower_lim: FilterLimit,
    upper_lim: FilterLimit,
    stats_mode: ComputationMode,
) -> anyhow::Result<()> {
    let mut num_genes_per_cell: Option<Vec<u32>> = None;
    let mut sum_genes_per_cell: Option<Vec<f64>> = None;

    if lower_lim.is_absolute() || upper_lim.is_absolute() {
        num_genes_per_cell = Some(crate::statistics::compute_n_genes(
            anndata,
            stats_mode.clone(),
        )?);
    }
    if lower_lim.is_relative() || upper_lim.is_relative() {
        sum_genes_per_cell = Some(crate::statistics::compute_sum_genes(
            anndata,
            stats_mode.clone(),
        )?);
    }

    let n_cells = anndata.n_obs();
    let filter_mask = get_removal_mask(
        num_genes_per_cell,
        sum_genes_per_cell,
        lower_lim,
        upper_lim,
        n_cells,
    )?;

    let keep_cells = Array1::from_vec(filter_mask);
    super::helpers::filter_anndata_inplace(anndata, Some(keep_cells.view()), None)?;

    Ok(())
}

pub fn remove_genes_inplace<B: Backend>(
    anndata: &mut AnnData<B>,
    lower_lim: FilterLimit,
    upper_lim: FilterLimit,
    stats_mode: ComputationMode,
) -> anyhow::Result<()> {
    let mut num_cells_per_gene: Option<Vec<u32>> = None;
    let mut sum_expr_per_gene: Option<Vec<f64>> = None;

    if lower_lim.is_absolute() || upper_lim.is_absolute() {
        num_cells_per_gene = Some(crate::statistics::compute_n_cells(
            anndata,
            stats_mode.clone(),
        )?);
    }
    if lower_lim.is_relative() || upper_lim.is_relative() {
        sum_expr_per_gene = Some(crate::statistics::compute_sum_cells(
            anndata,
            stats_mode.clone(),
        )?);
    }

    let n_genes = anndata.n_vars();

    let mask = get_removal_mask(
        num_cells_per_gene,
        sum_expr_per_gene,
        lower_lim,
        upper_lim,
        n_genes,
    )?;

    let keep_genes = Array1::from_vec(mask);
    super::helpers::filter_anndata_inplace(anndata, None, Some(keep_genes.view()))?;

    Ok(())
}

pub fn remove_cells<B: Backend, P: AsRef<Path>>(
    anndata: &AnnData<B>,
    lower_lim: FilterLimit,
    upper_lim: FilterLimit,
    stats_mode: ComputationMode,
    file_name: P,
    file_scope: FileScope,
) -> anyhow::Result<AnnData<H5>> {
    let mut num_genes_per_cell: Option<Vec<u32>> = None;
    let mut sum_genes_per_cell: Option<Vec<f64>> = None;

    if lower_lim.is_absolute() || upper_lim.is_absolute() {
        num_genes_per_cell = Some(crate::statistics::compute_n_genes(
            anndata,
            stats_mode.clone(),
        )?);
    }
    if lower_lim.is_relative() || upper_lim.is_relative() {
        sum_genes_per_cell = Some(crate::statistics::compute_sum_genes(
            anndata,
            stats_mode.clone(),
        )?);
    }

    let n_cells = anndata.n_obs();
    let filter_mask = get_removal_mask(
        num_genes_per_cell,
        sum_genes_per_cell,
        lower_lim,
        upper_lim,
        n_cells,
    )?;

    let keep_cells = Array1::from_vec(filter_mask);
    super::helpers::filter_anndata(
        anndata,
        Some(keep_cells.view()),
        None,
        file_name,
        file_scope,
    )
}

pub fn remove_genes<B: Backend, P: AsRef<Path>>(
    anndata: &mut AnnData<B>,
    lower_lim: FilterLimit,
    upper_lim: FilterLimit,
    stats_mode: ComputationMode,
    file_name: P,
    file_scope: FileScope,
) -> anyhow::Result<AnnData<H5>> {
    let mut num_cells_per_gene: Option<Vec<u32>> = None;
    let mut sum_expr_per_gene: Option<Vec<f64>> = None;

    if lower_lim.is_absolute() || upper_lim.is_absolute() {
        num_cells_per_gene = Some(crate::statistics::compute_n_cells(
            anndata,
            stats_mode.clone(),
        )?);
    }
    if lower_lim.is_relative() || upper_lim.is_relative() {
        sum_expr_per_gene = Some(crate::statistics::compute_sum_cells(
            anndata,
            stats_mode.clone(),
        )?);
    }

    let n_genes = anndata.n_vars();

    let mask = get_removal_mask(
        num_cells_per_gene,
        sum_expr_per_gene,
        lower_lim,
        upper_lim,
        n_genes,
    )?;

    let keep_genes = Array1::from_vec(mask);
    super::helpers::filter_anndata(
        anndata,
        None,
        Some(keep_genes.view()),
        file_name,
        file_scope,
    )
}


// pub fn remove_genes_and_cells_inplace<B: Backend>(
//     anndata: &mut AnnData<B>,
//     genes_lower_lim: FilterLimit,
//     genes_upper_lim: FilterLimit,
//     cells_lower_lim: FilterLimit,
//     cells_upper_lim: FilterLimit,
//     stats_mode: ComputationMode,
// ) -> anyhow::Result<()> {
//     let mut num_cells_per_gene: Option<Vec<u32>> = None;
//     let mut sum_expr_per_gene: Option<Vec<f64>> = None;
//     let mut num_genes_per_cell: Option<Vec<u32>> = None;
//     let mut sum_genes_per_cell: Option<Vec<f64>> = None;

//     if genes_lower_lim.is_absolute() || cells_lower_lim.is_absolute() {
//         num_cells_per_gene = Some(crate::statistics::compute_n_cells(
//             anndata,
//             stats_mode.clone(),
//         )?);
//     }
//     if genes_lower_lim.is_relative() || cells_lower_lim.is_relative() {
//         sum_expr_per_gene = Some(crate::statistics::compute_sum_cells(
//             anndata,
//             stats_mode.clone(),
//         )?);
//     }



//     let n_genes = anndata.n_vars();

//     let mask = get_removal_mask(
//         num_cells_per_gene,
//         sum_expr_per_gene,
//         lower_lim,
//         upper_lim,
//         n_genes,
//     )?;

//     let keep_genes = Array1::from_vec(mask);
//     super::helpers::filter_anndata_inplace(anndata, None, Some(keep_genes.view()))?;

//     Ok(())
// }