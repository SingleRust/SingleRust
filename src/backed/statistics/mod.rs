use anndata::{ArrayData, ArrayElem, ArrayElemOp, Backend};
use anyhow::bail;
use num_traits::{PrimInt, Unsigned, Zero};
use single_algebra::sparse::{MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance};
use std::ops::AddAssign;

use crate::{
    match_array_data_apply_function, match_array_data_apply_function_with_generics, ComputeMinMax,
    ComputeNonZero, ComputeSum, ComputeVariance,
};

impl<B: Backend> ComputeNonZero for ArrayElem<B> {
    fn nonzero_whole<T>(&self, direction: &single_algebra::Direction) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(x_data, nonzero_col)
            }
            single_algebra::Direction::ROW => match_array_data_apply_function!(x_data, nonzero_row),
        }
    }

    fn nonzero_chunk<T>(
        &self,
        direction: &single_algebra::Direction,
        reference: &mut [T],
    ) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        match direction {
            single_algebra::Direction::COLUMN => {
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(chunk, nonzero_col_chunk, reference)?;
                }
            }
            single_algebra::Direction::ROW => {
                for (chunk, start, end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(
                        chunk,
                        nonzero_row_chunk,
                        &mut reference[start..end]
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl<B: Backend> ComputeSum for ArrayElem<B> {
    fn sum_whole<T>(&self, direction: &single_algebra::Direction) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(x_data, sum_col)
            }
            single_algebra::Direction::ROW => match_array_data_apply_function!(x_data, sum_row),
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
        match direction {
            single_algebra::Direction::COLUMN => {
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(chunk, sum_col_chunk, reference)?;
                }
            }
            single_algebra::Direction::ROW => {
                for (chunk, start, end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(
                        chunk,
                        sum_row_chunk,
                        &mut reference[start..end]
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl<B: Backend> ComputeVariance for ArrayElem<B> {
    fn variance_whole<I, T>(&self, direction: &single_algebra::Direction) -> anyhow::Result<Vec<T>>
    where
        I: num_traits::PrimInt
            + num_traits::Unsigned
            + num_traits::Zero
            + std::ops::AddAssign
            + Into<T>,
        T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function_with_generics!(x_data, var_col, [I, T])
            }
            single_algebra::Direction::ROW => {
                match_array_data_apply_function_with_generics!(x_data, var_row, [I, T])
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
        match direction {
            single_algebra::Direction::COLUMN => {
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function_with_generics!(
                        chunk,
                        var_col_chunk,
                        [I, T],
                        reference
                    )?;
                }
            }
            single_algebra::Direction::ROW => {
                for (chunk, start, end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function_with_generics!(
                        chunk,
                        var_row_chunk,
                        [I, T],
                        &mut reference[start..end]
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl<B: Backend> ComputeMinMax for ArrayElem<B> {
    fn min_max_whole<T>(
        &self,
        direction: &single_algebra::Direction,
    ) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: num_traits::NumCast + Copy + PartialOrd + single_algebra::NumericOps,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            single_algebra::Direction::COLUMN => {
                match_array_data_apply_function!(x_data, min_max_col)
            }
            single_algebra::Direction::ROW => match_array_data_apply_function!(x_data, min_max_row),
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
        // Destructure the reference tuple into min and max vectors
        let (min_ref, max_ref) = reference;

        match direction {
            single_algebra::Direction::COLUMN => {
                // For columns, we need the full vectors since any chunk can affect any column
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(chunk, min_max_col_chunk, (min_ref, max_ref))?;
                }
            }
            single_algebra::Direction::ROW => {
                // For rows, only process the slices corresponding to rows in this chunk
                for (chunk, start, end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(
                        chunk,
                        min_max_col_chunk,
                        (&mut min_ref[start..end], &mut max_ref[start..end])
                    )?;
                }
            }
        }
        Ok(())
    }
}
