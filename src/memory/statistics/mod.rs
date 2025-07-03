use std::ops::{AddAssign, Deref};
pub mod structs;

use anndata_memory::IMArrayElement;
use anyhow::{bail, Ok};
use num_traits::{PrimInt, Unsigned, Zero};
use single_algebra::sparse::{MatrixMinMax, MatrixNTop, MatrixNonZero, MatrixSum, MatrixVariance};
use single_utilities::traits::NumericOps;
use single_utilities::types::Direction;

mod qc;
pub use qc::calculate_qc_metrics;
pub use qc::qc_metrics;

use crate::{
    match_array_data_apply_function, match_array_data_apply_function_with_generics,
    shared::statistics::{
        ComputeMinMax, ComputeNTop, ComputeNonZero, ComputeSum, ComputeTopSegmentProportions,
        ComputeVariance,
    },
};

impl ComputeNonZero for IMArrayElement {
    fn nonzero_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, nonzero_col)
            }
            Direction::ROW => match_array_data_apply_function!(data, nonzero_row),
        }
    }

    fn nonzero_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, nonzero_col_chunk, reference)
            }
            Direction::ROW => {
                match_array_data_apply_function!(data, nonzero_row_chunk, reference)
            }
        }
    }

    fn nonzero_whole_masked<T>(
        &self,
        direction: &Direction,
        mask: &[bool],
    ) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, nonzero_col_masked, mask)
            }
            Direction::ROW => {
                match_array_data_apply_function!(data, nonzero_row_masked, mask)
            }
        }
    }
}

impl ComputeSum for IMArrayElement {
    fn sum_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, sum_col)
            }
            Direction::ROW => match_array_data_apply_function!(data, sum_row),
        }
    }

    fn sum_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, sum_col_chunk, reference)
            }
            Direction::ROW => {
                match_array_data_apply_function!(data, sum_row_chunk, reference)
            }
        }
    }

    fn sum_whole_masked<T>(&self, direction: &Direction, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, sum_col_masked, mask)
            }
            Direction::ROW => {
                match_array_data_apply_function!(data, sum_row_masked, mask)
            }
        }
    }
}

impl ComputeVariance for IMArrayElement {
    fn variance_whole<I, T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        I: num_traits::PrimInt
            + num_traits::Unsigned
            + num_traits::Zero
            + std::ops::AddAssign
            + Into<T> + Send + Sync,
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + Send + Sync,
    {
        let read_guard = self.0.read_inner();

        let data = read_guard.deref();

        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function_with_generics!(data, var_col, [I, T])
            }
            Direction::ROW => {
                match_array_data_apply_function_with_generics!(data, var_row, [I, T])
            }
        }
    }

    fn variance_chunk<I, T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: num_traits::PrimInt
            + num_traits::Unsigned
            + num_traits::Zero
            + std::ops::AddAssign
            + Into<T> + Send + Sync,
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + Send + Sync,
    {
        let read_guard = self.0.read_inner();

        let data = read_guard.deref();

        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function_with_generics!(
                    data,
                    var_col_chunk,
                    [I, T],
                    reference
                )
            }
            Direction::ROW => {
                match_array_data_apply_function_with_generics!(
                    data,
                    var_row_chunk,
                    [I, T],
                    reference
                )
            }
        }
    }
}

impl ComputeMinMax for IMArrayElement {
    fn min_max_whole<T>(&self, direction: &Direction) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: num_traits::NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let read_guard = self.0.read_inner();

        let data = read_guard.deref();

        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, min_max_col)
            }
            Direction::ROW => match_array_data_apply_function!(data, min_max_row),
        }
    }

    fn min_max_chunk<T>(
        &self,
        direction: &Direction,
        reference: (&mut Vec<T>, &mut Vec<T>),
    ) -> anyhow::Result<()>
    where
        T: num_traits::NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let read_guard = self.0.read_inner();

        let data = read_guard.deref();

        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(
                    data,
                    min_max_col_chunk,
                    (reference.0, reference.1)
                )
            }
            Direction::ROW => {
                match_array_data_apply_function!(
                    data,
                    min_max_row_chunk,
                    (reference.0, reference.1)
                )
            }
        }
    }
}

impl ComputeNTop for IMArrayElement {
    fn n_top_whole<T>(&self, direction: &Direction, n: usize) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(data, sum_row_n_top, n)
            }
            Direction::ROW => {
                match_array_data_apply_function!(data, sum_row_n_top, n)
            }
        }
    }
}

impl ComputeTopSegmentProportions for IMArrayElement {
    fn top_segment_proportions(
        &self,
        direction: &Direction,
        ns: &[usize],
    ) -> anyhow::Result<ndarray::Array2<f64>> {
        let shape = self.get_shape()?;
        let (n_items, n_features) = match direction {
            Direction::ROW => (shape[0], shape[1]),
            Direction::COLUMN => (shape[1], shape[0]),
        };

        for &n in ns {
            if n > n_features {
                return Err(anyhow::anyhow!(
                    "Requested top {} features but only {} available",
                    n,
                    n_features
                ));
            }
        }

        let totals: Vec<f64> = self.sum_whole(direction)?;

        const BATCH_SIZE: usize = 1000;
        let mut proportions = ndarray::Array2::<f64>::zeros((n_items, ns.len()));

        let mut unique_ns: Vec<usize> = ns.to_vec();
        unique_ns.sort_unstable();
        unique_ns.dedup();

        let mut n_to_indices: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, &n) in ns.iter().enumerate() {
            n_to_indices.entry(n).or_default().push(idx);
        }

        for &n in &unique_ns {
            let top_values: Vec<f64> = self.n_top_whole(direction, n)?;

            for item_idx in 0..n_items {
                let total = totals[item_idx];
                if total > 0.0 {
                    let start_idx = item_idx * n;
                    let end_idx = start_idx + n;
                    let sum_top_n: f64 = top_values[start_idx..end_idx].iter().sum();
                    let proportion = sum_top_n / total;

                    for &ns_idx in &n_to_indices[&n] {
                        proportions[[item_idx, ns_idx]] = proportion;
                    }
                }
            }
        }

        Ok(proportions)
    }
}

/*
pub fn qc_vars_inplace<I, T>(adata: &IMAnnData) -> anyhow::Result<()>
where
    I: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + From<I>,
{
    let data = compute_qc_variables::<I, T>(adata)?;

    let mut obs_df = adata.obs().get_data();
    let mut var_df = adata.var().get_data();

    let num_cell_series = Series::from_vec("num_genes_per_cell".into(), data.num_per_cell);
    let num_genes_series = Series::from_vec("num_cells_per_gene".into(), data.num_per_gene);
    let expr_per_gene_series = Series::from_vec("sum_expr_per_gene".into(), data.expr_per_gene);
    let expr_per_cell_series = Series::from_vec("sum_expr_per_cell".into(), data.expr_per_cell);
    let var_per_gene_series = Series::from_vec("var_expr_per_gene".into(), data.variance_per_gene);
    let var_per_cell_series = Series::from_vec("var_expr_per_cell".into(), data.variance_per_cell);
    let std_dev_per_gene_series =
        Series::from_vec("std_dev_per_gene".into(), data.std_dev_per_gene);
    let std_dev_per_cell_series =
        Series::from_vec("std_dev_per_cell".into(), data.std_dev_per_cell);

    obs_df.with_column(num_cell_series)?;
    obs_df.with_column(expr_per_cell_series)?;
    obs_df.with_column(var_per_cell_series)?;
    obs_df.with_column(std_dev_per_cell_series)?;

    var_df.with_column(num_genes_series)?;
    var_df.with_column(expr_per_gene_series)?;
    var_df.with_column(var_per_gene_series)?;
    var_df.with_column(std_dev_per_gene_series)?;

    adata.obs().set_data(obs_df)?;
    adata.var().set_data(var_df)?;

    Ok(())
}
*/
