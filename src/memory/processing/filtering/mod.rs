use anndata_memory::IMAnnData;
use single_algebra::Direction;

use crate::shared::statistics::{ComputeNonZero, ComputeSum};

/// Filtering cells by different citeria
/// TODO: have to overhaul typing here with I and T respectively for each filtering step
pub fn mark_filter_cells<I, T>(
    anndata: &IMAnnData,
    min_genes: Option<I>,
    max_genes: Option<I>,
    min_counts: Option<T>,
    max_counts: Option<T>,
    min_fraction: Option<T>,
    max_fraction: Option<T>
) -> anyhow::Result<Vec<bool>>
where
    I: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign + Into<T>,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
{
    let mut keep_cells = vec![true; anndata.n_obs()];
    let x_elem = anndata.x().clone();

    if min_genes.is_some() || max_genes.is_some() || min_fraction.is_some() || max_fraction.is_some() {
        let genes_per_cell = x_elem.nonzero_whole::<I>(&Direction::ROW)?;

        if let Some(min_genes_threshold) = min_genes {
            let new_filter: Vec<bool> = genes_per_cell.iter()
                .map(|&x| x >= min_genes_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        if let Some(max_genes_threshold) = max_genes {
            let new_filter: Vec<bool> = genes_per_cell.iter()
                .map(|&x| x <= max_genes_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        let n_genes = T::from(anndata.n_vars()).unwrap();

        if let Some(min_frac) = min_fraction {
            let new_filter: Vec<bool> = genes_per_cell.iter()
                .map(|&x| (T::from(x).unwrap() / n_genes) >= min_frac)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        if let Some(max_frac) = max_fraction {
            let new_filter: Vec<bool> = genes_per_cell.iter()
                .map(|&x| (T::from(x).unwrap() / n_genes) <= max_frac)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }
    }

    if min_counts.is_some() || max_counts.is_some() {
        let counts_per_cell = x_elem.sum_whole::<T>(&Direction::ROW)?;

        if let Some(min_counts_threshold) = min_counts {
            let new_filter: Vec<bool> = counts_per_cell.iter()
                .map(|&x| x >= min_counts_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        if let Some(max_counts_threshold) = max_counts {
            let new_filter: Vec<bool> = counts_per_cell.iter()
                .map(|&x| x <= max_counts_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }
    }

    Ok(keep_cells)
}

pub fn mark_filter_genes<I, T>(
    anndata: &IMAnnData,
    min_cells: Option<I>,
    max_cells: Option<I>,
    min_counts: Option<T>,
    max_counts: Option<T>,
    min_fraction: Option<T>,
    max_fraction: Option<T>
) -> anyhow::Result<Vec<bool>>
where
    I: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign + Into<T>,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
{
    let mut keep_genes = vec![true; anndata.n_vars()];
    let x_elem = anndata.x().clone();

    if min_cells.is_some() || max_cells.is_some() || min_fraction.is_some() || max_fraction.is_some() {
        let cells_per_gene = x_elem.nonzero_whole::<I>(&Direction::COLUMN)?;

        if let Some(min_cells_threshold) = min_cells {
            let new_filter: Vec<bool> = cells_per_gene.iter()
                .map(|&x| x >= min_cells_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        if let Some(max_cells_threshold) = max_cells {
            let new_filter: Vec<bool> = cells_per_gene.iter()
                .map(|&x| x <= max_cells_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        let n_cells = T::from(anndata.n_obs()).unwrap();

        if let Some(min_frac) = min_fraction {
            let new_filter: Vec<bool> = cells_per_gene.iter()
                .map(|&x| (T::from(x).unwrap() / n_cells) >= min_frac)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        if let Some(max_frac) = max_fraction {
            let new_filter: Vec<bool> = cells_per_gene.iter()
                .map(|&x| (T::from(x).unwrap() / n_cells) <= max_frac)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }
    }

    if min_counts.is_some() || max_counts.is_some() {
        let counts_per_gene = x_elem.sum_whole::<T>(&Direction::COLUMN)?;

        if let Some(min_counts_threshold) = min_counts {
            let new_filter: Vec<bool> = counts_per_gene.iter()
                .map(|&x| x >= min_counts_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        if let Some(max_counts_threshold) = max_counts {
            let new_filter: Vec<bool> = counts_per_gene.iter()
                .map(|&x| x <= max_counts_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }
    }

    Ok(keep_genes)
}

fn combine_filters(filter1: &[bool], filter2: &[bool]) -> Vec<bool> {
    filter1.iter()
        .zip(filter2.iter())
        .map(|(&x, &y)| x && y)
        .collect()
}


