use crate::shared::statistics::{ComputeNonZero, ComputeSum};
use anndata_memory::IMAnnData;
use single_utilities::types::Direction;

/// Filter cells based on various quality control metrics.
///
/// This function creates a boolean mask indicating which cells pass all specified filtering criteria.
/// The filtering is performed based on the number of genes expressed per cell and/or the total counts per cell.
///
/// # Arguments
///
/// * `anndata` - Reference to an IMAnnData object containing the single-cell data
/// * `min_genes` - Optional minimum number of genes expressed required for a cell to pass filtering
/// * `max_genes` - Optional maximum number of genes expressed allowed for a cell to pass filtering
/// * `min_counts` - Optional minimum count total required for a cell to pass filtering
/// * `max_counts` - Optional maximum count total allowed for a cell to pass filtering
/// * `min_fraction` - Optional minimum fraction of total genes that must be expressed in a cell
/// * `max_fraction` - Optional maximum fraction of total genes that can be expressed in a cell
///
/// # Type Parameters
///
/// * `I` - Integer type for gene counts, must be unsigned and implement necessary numeric traits
/// * `T` - Floating point type for counts and fractions, must implement necessary numeric traits
///
/// # Returns
///
/// Returns `Result<Vec<bool>>` where the vector contains `true` for cells that pass all filters
/// and `false` for cells that fail any filter criterion.
///
/// # Examples
///
/// ```
/// let filtered_cells = mark_filter_cells(
///     &anndata,
///     Some(200),     // Require at least 200 genes expressed
///     Some(5000),    // Allow at most 5000 genes expressed
///     Some(1000.0),  // Require at least 1000 total counts
///     None,          // No maximum count threshold
///     Some(0.001),   // Require at least 0.1% of genes expressed
///     Some(0.1),     // Allow at most 10% of genes expressed
/// )?;
/// ```
///
/// # Notes
///
/// - If a filter criterion is None, that particular filter will not be applied
/// - All specified criteria must be met for a cell to pass filtering (logical AND)
/// - Gene counts are calculated as number of non-zero entries per cell
/// - Total counts are calculated as sum of all entries per cell
/// - Fractions are calculated relative to the total number of genes in the dataset
pub fn mark_filter_cells<I, T>(
    anndata: &IMAnnData,
    min_genes: Option<I>,
    max_genes: Option<I>,
    min_counts: Option<T>,
    max_counts: Option<T>,
    min_fraction: Option<T>,
    max_fraction: Option<T>,
) -> anyhow::Result<Vec<bool>>
where
    I: num_traits::PrimInt
        + num_traits::Unsigned
        + num_traits::Zero
        + std::ops::AddAssign
        + Into<T>
        + Send
        + Sync,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + Send + Sync,
{
    let mut keep_cells = vec![true; anndata.n_obs()];
    let x_elem = anndata.x().clone();

    if min_genes.is_some()
        || max_genes.is_some()
        || min_fraction.is_some()
        || max_fraction.is_some()
    {
        let genes_per_cell = x_elem.nonzero_whole::<I>(&Direction::ROW)?;

        if let Some(min_genes_threshold) = min_genes {
            let new_filter: Vec<bool> = genes_per_cell
                .iter()
                .map(|&x| x >= min_genes_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        if let Some(max_genes_threshold) = max_genes {
            let new_filter: Vec<bool> = genes_per_cell
                .iter()
                .map(|&x| x <= max_genes_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        let n_genes = T::from(anndata.n_vars()).unwrap();

        if let Some(min_frac) = min_fraction {
            let new_filter: Vec<bool> = genes_per_cell
                .iter()
                .map(|&x| (T::from(x).unwrap() / n_genes) >= min_frac)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        if let Some(max_frac) = max_fraction {
            let new_filter: Vec<bool> = genes_per_cell
                .iter()
                .map(|&x| (T::from(x).unwrap() / n_genes) <= max_frac)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }
    }

    if min_counts.is_some() || max_counts.is_some() {
        let counts_per_cell = x_elem.sum_whole::<T>(&Direction::ROW)?;

        if let Some(min_counts_threshold) = min_counts {
            let new_filter: Vec<bool> = counts_per_cell
                .iter()
                .map(|&x| x >= min_counts_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }

        if let Some(max_counts_threshold) = max_counts {
            let new_filter: Vec<bool> = counts_per_cell
                .iter()
                .map(|&x| x <= max_counts_threshold)
                .collect();
            keep_cells = combine_filters(&keep_cells, &new_filter);
        }
    }

    Ok(keep_cells)
}

/// Filter genes based on various quality control metrics.
///
/// This function creates a boolean mask indicating which genes pass all specified filtering criteria.
/// The filtering is performed based on the number of cells expressing each gene and/or the total counts per gene.
///
/// # Arguments
///
/// * `anndata` - Reference to an IMAnnData object containing the single-cell data
/// * `min_cells` - Optional minimum number of cells expressing a gene required for it to pass filtering
/// * `max_cells` - Optional maximum number of cells expressing a gene allowed for it to pass filtering
/// * `min_counts` - Optional minimum count total required for a gene to pass filtering
/// * `max_counts` - Optional maximum count total allowed for a gene to pass filtering
/// * `min_fraction` - Optional minimum fraction of total cells that must express a gene
/// * `max_fraction` - Optional maximum fraction of total cells that can express a gene
///
/// # Type Parameters
///
/// * `I` - Integer type for cell counts, must be unsigned and implement necessary numeric traits
/// * `T` - Floating point type for counts and fractions, must implement necessary numeric traits
///
/// # Returns
///
/// Returns `Result<Vec<bool>>` where the vector contains `true` for genes that pass all filters
/// and `false` for genes that fail any filter criterion.
///
/// # Examples
///
/// ```
/// let filtered_genes = mark_filter_genes(
///     &anndata,
///     Some(3),       // Require expression in at least 3 cells
///     None,          // No maximum number of cells threshold
///     Some(10.0),    // Require at least 10 total counts
///     Some(1e5),     // Allow at most 100,000 total counts
///     Some(0.001),   // Require expression in at least 0.1% of cells
///     Some(0.9),     // Allow expression in at most 90% of cells
/// )?;
/// ```
///
/// # Notes
///
/// - If a filter criterion is None, that particular filter will not be applied
/// - All specified criteria must be met for a gene to pass filtering (logical AND)
/// - Cell counts are calculated as number of non-zero entries per gene
/// - Total counts are calculated as sum of all entries per gene
/// - Fractions are calculated relative to the total number of cells in the dataset
pub fn mark_filter_genes<I, T>(
    anndata: &IMAnnData,
    min_cells: Option<I>,
    max_cells: Option<I>,
    min_counts: Option<T>,
    max_counts: Option<T>,
    min_fraction: Option<T>,
    max_fraction: Option<T>,
) -> anyhow::Result<Vec<bool>>
where
    I: num_traits::PrimInt
        + num_traits::Unsigned
        + num_traits::Zero
        + std::ops::AddAssign
        + Into<T>
        + Send
        + Sync,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + Send + Sync,
{
    let mut keep_genes = vec![true; anndata.n_vars()];
    let x_elem = anndata.x();

    if min_cells.is_some()
        || max_cells.is_some()
        || min_fraction.is_some()
        || max_fraction.is_some()
    {
        let cells_per_gene = x_elem.nonzero_whole::<I>(&Direction::COLUMN)?;

        if let Some(min_cells_threshold) = min_cells {
            let new_filter: Vec<bool> = cells_per_gene
                .iter()
                .map(|&x| x >= min_cells_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        if let Some(max_cells_threshold) = max_cells {
            let new_filter: Vec<bool> = cells_per_gene
                .iter()
                .map(|&x| x <= max_cells_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        let n_cells = T::from(anndata.n_obs()).unwrap();

        if let Some(min_frac) = min_fraction {
            let new_filter: Vec<bool> = cells_per_gene
                .iter()
                .map(|&x| (T::from(x).unwrap() / n_cells) >= min_frac)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        if let Some(max_frac) = max_fraction {
            let new_filter: Vec<bool> = cells_per_gene
                .iter()
                .map(|&x| (T::from(x).unwrap() / n_cells) <= max_frac)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }
    }

    if min_counts.is_some() || max_counts.is_some() {
        let counts_per_gene = x_elem.sum_whole::<T>(&Direction::COLUMN)?;

        if let Some(min_counts_threshold) = min_counts {
            let new_filter: Vec<bool> = counts_per_gene
                .iter()
                .map(|&x| x >= min_counts_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }

        if let Some(max_counts_threshold) = max_counts {
            let new_filter: Vec<bool> = counts_per_gene
                .iter()
                .map(|&x| x <= max_counts_threshold)
                .collect();
            keep_genes = combine_filters(&keep_genes, &new_filter);
        }
    }

    Ok(keep_genes)
}

fn combine_filters(filter1: &[bool], filter2: &[bool]) -> Vec<bool> {
    filter1
        .iter()
        .zip(filter2.iter())
        .map(|(&x, &y)| x && y)
        .collect()
}
