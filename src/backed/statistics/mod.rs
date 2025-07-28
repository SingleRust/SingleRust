use crate::{
    match_array_data_apply_function, match_array_data_apply_function_with_generics, ComputeMinMax,
    ComputeNonZero, ComputeSum, ComputeVariance,
};
use anndata::{ArrayData, ArrayElem, ArrayElemOp, Backend};
use anyhow::bail;
use num_traits::{PrimInt, Unsigned, Zero};
use single_algebra::sparse::{MatrixMinMax, MatrixNonZero, MatrixSum, MatrixVariance};
use single_utilities::traits::NumericOps;
use single_utilities::types::Direction;
use std::ops::AddAssign;

impl<B: Backend> ComputeNonZero for ArrayElem<B> {
    fn nonzero_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(x_data, nonzero_col)
            }
            Direction::ROW => match_array_data_apply_function!(x_data, nonzero_row),
        }
    }

    fn nonzero_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync,
    {
        match direction {
            Direction::COLUMN => {
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(chunk, nonzero_col_chunk, reference)?;
                }
            }
            Direction::ROW => {
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

    fn nonzero_whole_masked<T>(
        &self,
        direction: &Direction,
        mask: &[bool],
    ) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign,
    {
        todo!()
    }
}

impl<B: Backend> ComputeSum for ArrayElem<B> {
    fn sum_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(x_data, sum_col)
            }
            Direction::ROW => match_array_data_apply_function!(x_data, sum_row),
        }
    }

    fn sum_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: num_traits::Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        match direction {
            Direction::COLUMN => {
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(chunk, sum_col_chunk, reference)?;
                }
            }
            Direction::ROW => {
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

    fn sum_whole_masked<T>(&self, direction: &Direction, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: num_traits::Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        todo!()
    }
}

impl<B: Backend> ComputeVariance for ArrayElem<B> {
    fn variance_whole<I, T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: num_traits::Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function_with_generics!(x_data, var_col, [I, T])
            }
            Direction::ROW => {
                match_array_data_apply_function_with_generics!(x_data, var_row, [I, T])
            }
        }
    }

    fn variance_chunk<I, T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: num_traits::Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync,
    {
        match direction {
            Direction::COLUMN => {
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function_with_generics!(
                        chunk,
                        var_col_chunk,
                        [I, T],
                        reference
                    )?;
                }
            }
            Direction::ROW => {
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
    fn min_max_whole<T>(&self, direction: &Direction) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: num_traits::NumCast + Copy + PartialOrd + NumericOps + Send + Sync,
    {
        let x_data = self.get::<ArrayData>()?.expect("X matrix not found!");
        match direction {
            Direction::COLUMN => {
                match_array_data_apply_function!(x_data, min_max_col)
            }
            Direction::ROW => match_array_data_apply_function!(x_data, min_max_row),
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
        // Destructure the reference tuple into min and max vectors
        let (min_ref, max_ref) = reference;

        match direction {
            Direction::COLUMN => {
                // For columns, we need the full vectors since any chunk can affect any column
                for (chunk, _start, _end) in self.iter::<ArrayData>(1000) {
                    match_array_data_apply_function!(chunk, min_max_col_chunk, (min_ref, max_ref))?;
                }
            }
            Direction::ROW => {
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
