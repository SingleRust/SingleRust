use std::ops::{AddAssign, Deref};
pub mod structs;

use anndata_memory::{IMAnnData, IMArrayElement};
use anyhow::bail;
use num_traits::{PrimInt, Unsigned, Zero};
use single_algebra::sparse::{MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance};
use single_algebra::Direction;
use structs::StatisticsContainer;

use crate::{
    match_array_data_apply_function, match_array_data_apply_function_with_generics,
    shared::statistics::{ComputeMinMax, ComputeNonZero, ComputeSum, ComputeVariance},
};

impl ComputeNonZero for IMArrayElement {
    fn nonzero_whole<T>(&self, direction: &single_algebra::Direction) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(data, nonzero_col)
            }
            single_algebra::Direction::ROW => match_array_data_apply_function!(data, nonzero_row),
        }
    }

    fn nonzero_chunk<T>(
        &self,
        direction: &single_algebra::Direction,
        reference: &mut [T],
    ) -> anyhow::Result<()>
    where
        T: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(data, nonzero_col_chunk, reference)
            }
            single_algebra::Direction::ROW => {
                match_array_data_apply_function!(data, nonzero_row_chunk, reference)
            }
        }
    }

    // #[cfg(feature = "simba")]
    // fn simba_nonzero_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T::Element>>
    // where
    //     T: simba::simd::SimdValue + simba::simd::PrimitiveSimdValue,
    //     T::Element: PrimInt + Unsigned + Zero + AddAssign,
    // {
    //     let read_guard = self.0.read_inner();
    //     let data = read_guard.deref();
    //     match direction {
    //         single_algebra::Direction::COLUMN => {
    //             match_array_data_apply_function_with_generics!(data, simba_nonzero_col, [T])
    //         }
    //         single_algebra::Direction::ROW => {
    //             match_array_data_apply_function_with_generics!(data, simba_nonzero_row, [T])
    //         }
    //     }
    // }
}

impl ComputeSum for IMArrayElement {
    fn sum_whole<T>(&self, direction: &single_algebra::Direction) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(data, sum_col)
            }
            single_algebra::Direction::ROW => match_array_data_apply_function!(data, sum_row),
        }
    }

    fn sum_chunk<T>(
        &self,
        direction: &single_algebra::Direction,
        reference: &mut [T],
    ) -> anyhow::Result<()>
    where
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
    {
        let read_guard = self.0.read_inner();
        let data = read_guard.deref();
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(data, sum_col_chunk, reference)
            }
            single_algebra::Direction::ROW => {
                match_array_data_apply_function!(data, sum_row_chunk, reference)
            }
        }
    }
}

impl ComputeVariance for IMArrayElement {
    fn variance_whole<I, T>(&self, direction: &single_algebra::Direction) -> anyhow::Result<Vec<T>>
    where
        I: num_traits::PrimInt
            + num_traits::Unsigned
            + num_traits::Zero
            + std::ops::AddAssign
            + Into<T>,
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
    {
        let read_guard = self.0.read_inner(); // establish a read guard

        let data = read_guard.deref();

        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function_with_generics!(data, var_col, [I, T])
            }
            single_algebra::Direction::ROW => {
                match_array_data_apply_function_with_generics!(data, var_row, [I, T])
            }
        }
    }

    fn variance_chunk<I, T>(
        &self,
        direction: &single_algebra::Direction,
        reference: &mut [T],
    ) -> anyhow::Result<()>
    where
        I: num_traits::PrimInt
            + num_traits::Unsigned
            + num_traits::Zero
            + std::ops::AddAssign
            + Into<T>,
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
    {
        let read_guard = self.0.read_inner(); // establish a read guard

        let data = read_guard.deref();

        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function_with_generics!(
                    data,
                    var_col_chunk,
                    [I, T],
                    reference
                )
            }
            single_algebra::Direction::ROW => {
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
    fn min_max_whole<T>(
        &self,
        direction: &single_algebra::Direction,
    ) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: num_traits::NumCast + Copy + PartialOrd + single_algebra::NumericOps,
    {
        let read_guard = self.0.read_inner(); // establish a read guard

        let data = read_guard.deref();

        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(data, min_max_col)
            }
            single_algebra::Direction::ROW => match_array_data_apply_function!(data, min_max_row),
        }
    }

    fn min_max_chunk<T>(
        &self,
        direction: &single_algebra::Direction,
        reference: (&mut Vec<T>, &mut Vec<T>),
    ) -> anyhow::Result<()>
    where
        T: num_traits::NumCast + Copy + PartialOrd + single_algebra::NumericOps,
    {
        let read_guard = self.0.read_inner(); // establish a read guard

        let data = read_guard.deref();

        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(
                    data,
                    min_max_col_chunk,
                    (reference.0, reference.1)
                )
            }
            single_algebra::Direction::ROW => {
                match_array_data_apply_function!(
                    data,
                    min_max_row_chunk,
                    (reference.0, reference.1)
                )
            }
        }
    }
}

pub fn compute_qc_variables<I, T>(adata: &IMAnnData) -> anyhow::Result<StatisticsContainer<I, T>>
where
    I: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + From<I>,
{
    let x = adata.x();

    /* let n_per_gene = crate::shared::statistics::number::whole(data, Direction::Column)?;
    let n_per_cell = crate::shared::statistics::number::whole(data, Direction::Row)?;
    let sum_per_gene = crate::shared::statistics::sum::whole(data, Direction::Column)?;
    let sum_per_cell = crate::shared::statistics::sum::whole(data, Direction::Row)?;
    let var_per_gene = crate::shared::statistics::variance::whole(data, Direction::Column)?;
    let var_per_cell = crate::shared::statistics::variance::whole(data, Direction::Row)?;
    let std_dev_per_gene = crate::shared::statistics::stddev::whole(data, Direction::Column)?;
    let std_dev_per_cell = crate::shared::statistics::stddev::whole(data, Direction::Row)?;
    */

    let n_per_gene: Vec<I> = x.nonzero_whole(&single_algebra::Direction::COLUMN)?;
    let n_per_cell: Vec<I> = x.nonzero_whole(&single_algebra::Direction::ROW)?;
    let sum_per_gene: Vec<T> = x.sum_whole(&single_algebra::Direction::COLUMN)?;
    let sum_per_cell: Vec<T> = x.sum_whole(&single_algebra::Direction::ROW)?;
    let var_per_gene: Vec<T> = x.variance_whole::<I, T>(&single_algebra::Direction::COLUMN)?;
    let var_per_cell: Vec<T> = x.variance_whole::<I, T>(&single_algebra::Direction::ROW)?;
    let std_dev_per_gene: Vec<T> = var_per_gene.iter().map(|x| x.sqrt()).collect();
    let std_dev_per_cell: Vec<T> = var_per_cell.iter().map(|x| x.sqrt()).collect();

    Ok(StatisticsContainer {
        num_per_cell: n_per_cell,
        num_per_gene: n_per_gene,
        expr_per_gene: sum_per_gene,
        expr_per_cell: sum_per_cell,
        variance_per_gene: var_per_gene,
        variance_per_cell: var_per_cell,
        std_dev_per_cell,
        std_dev_per_gene,
    })
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
