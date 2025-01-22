use anndata_memory::IMAnnData;
use num_traits::NumCast;
use single_algebra::Direction;

use crate::FlexValue;

use crate::shared::statistics::{ComputeNonZero, ComputeSum};

/// Filtering cells by different citeria
/// TODO: have to overhaul typing here with I and T respectively for each filtering step
/// For now the FlexValue is used for that.
/// True are all cells to be kept, false are all cells to be removed
pub fn mark_filter_cells<I, T>(
    anndata: &IMAnnData,
    min_num_genes: FlexValue,
    max_num_genes: FlexValue,
    min_num_gene_expression: FlexValue,
    max_num_gene_expression: FlexValue,
) -> anyhow::Result<Vec<bool>>
// Added Result return type
where
    I: num_traits::PrimInt
        + num_traits::Unsigned
        + num_traits::Zero
        + std::ops::AddAssign
        + Into<T>,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
{
    let num_genes = anndata.n_vars() as f32;
    let mut filter_cells = vec![true; anndata.n_obs()];

    let x_elem = anndata.x().clone();

    if (!min_num_genes.is_none() || !max_num_genes.is_none()) {
        let non_zero_counts = x_elem.nonzero_whole::<I>(&Direction::ROW)?;
        let num_genes_t = <T as NumCast>::from(num_genes).unwrap();
        let non_zero_counts_relative: Vec<T> = non_zero_counts
            .iter()
            .map(|&x| x.into() / num_genes_t)
            .collect();

        let mut combined_filters = vec![true; filter_cells.len()];

        match min_num_genes {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = non_zero_counts
                    .iter()
                    .map(|&x| x.into() >= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                let value_t = <T as NumCast>::from(value).unwrap();
                let filters: Vec<bool> = non_zero_counts_relative
                    .iter()
                    .map(|&x| x >= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::None => {}
        }

        match max_num_genes {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = non_zero_counts
                    .iter()
                    .map(|&x| x.into() <= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                let value_t = <T as NumCast>::from(value).unwrap();
                let filters: Vec<bool> = non_zero_counts_relative
                    .iter()
                    .map(|&x| x <= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::None => {}
        }

        filter_cells = filter_cells
            .iter()
            .zip(combined_filters.iter())
            .map(|(&x, &y)| x && y)
            .collect();
    }

    if !min_num_gene_expression.is_none() || !max_num_gene_expression.is_none() {
        let sum_counts = x_elem.sum_whole::<T>(&Direction::ROW)?;
        let mut combined_filters = vec![true; filter_cells.len()];

        match min_num_gene_expression {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = sum_counts.iter().map(|&x| x >= value_t).collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                todo!("This function still needs to be implemented! FUNCTION: filter_cells:filtering_relative:min_num:relative")
            }
            FlexValue::None => {}
        }

        match max_num_gene_expression {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = sum_counts.iter().map(|&x| x <= value_t).collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                todo!("This function still needs to be implemented! FUNCTION: filter_cells:filtering_relative:max_num:relative")
            }
            FlexValue::None => {}
        }

        filter_cells = filter_cells
            .iter()
            .zip(combined_filters.iter())
            .map(|(&x, &y)| x && y)
            .collect();
    }

    Ok(filter_cells)
}

pub fn mark_filter_genes<I, T>(
    anndata: &IMAnnData,
    min_num_cells: FlexValue, // number of non-zero cells per gene, either percentage or absolute
    max_num_cells: FlexValue, // number of non-zero cells per gene, either percentage or absolute
    min_num_cell_expression: FlexValue, // total expression per gene
    max_num_cell_expression: FlexValue, // total expression per gene
) -> anyhow::Result<Vec<bool>>
where
    I: num_traits::PrimInt
        + num_traits::Unsigned
        + num_traits::Zero
        + std::ops::AddAssign
        + Into<T>,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
{
    let num_cells = anndata.n_obs() as f32;
    let mut filter_genes = vec![true; anndata.n_vars()];

    let x_elem = anndata.x().clone(); // only shallow clone here!

    if min_num_cells.is_some() || max_num_cells.is_some() {
        let non_zero_counts = x_elem.nonzero_whole::<I>(&Direction::COLUMN)?;
        let num_cells_t = <T as NumCast>::from(num_cells).unwrap();
        let non_zero_counts_relative: Vec<T> = non_zero_counts
            .iter()
            .map(|&x| x.into() / num_cells_t)
            .collect();

        let mut combined_filters = vec![true; filter_genes.len()];

        match min_num_cells {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = non_zero_counts
                    .iter()
                    .map(|&x| x.into() >= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                let value_t = <T as NumCast>::from(value).unwrap();
                let filters: Vec<bool> = non_zero_counts_relative
                    .iter()
                    .map(|&x| x >= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::None => {}
        }

        match max_num_cells {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = non_zero_counts
                    .iter()
                    .map(|&x| x.into() <= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                let value_t = <T as NumCast>::from(value).unwrap();
                let filters: Vec<bool> = non_zero_counts_relative
                    .iter()
                    .map(|&x| x <= value_t)
                    .collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::None => {}
        }

        filter_genes = filter_genes
            .iter()
            .zip(combined_filters.iter())
            .map(|(&x, &y)| x && y)
            .collect();
    }

    if min_num_cell_expression.is_some() || max_num_cell_expression.is_some() {
        let sum_counts = x_elem.sum_whole::<T>(&Direction::COLUMN)?;
        let mut combined_filters = vec![true; filter_genes.len()];

        match min_num_cell_expression {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = sum_counts.iter().map(|&x| x >= value_t).collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                todo!("This function still needs to be implemented! FUNCTION: filter_genes:filtering_relative:min_num:relative")
            }
            FlexValue::None => {}
        }

        match max_num_cell_expression {
            FlexValue::Absolute(value) => {
                let value_t = <T as NumCast>::from(value.round()).unwrap();
                let filters: Vec<bool> = sum_counts.iter().map(|&x| x <= value_t).collect();
                combined_filters = combined_filters
                    .iter()
                    .zip(filters.iter())
                    .map(|(&x, &y)| x && y)
                    .collect();
            }
            FlexValue::Relative(value) => {
                todo!("This function still needs to be implemented! FUNCTION: filter_genes:filtering_relative:max_num:relative")
            }
            FlexValue::None => {}
        }

        filter_genes = filter_genes
            .iter()
            .zip(combined_filters.iter())
            .map(|(&x, &y)| x && y)
            .collect();
    }

    Ok(filter_genes)
}
