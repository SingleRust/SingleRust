use std::ops::AddAssign;

use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use single_utilities::traits::NumericOps;
use single_utilities::types::Direction;

pub trait ComputeNonZero {
    fn nonzero_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;

    fn nonzero_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign;
}

pub trait ComputeSum {
    fn sum_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn sum_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;
}

pub trait ComputeVariance {
    fn variance_whole<I, T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;

    fn variance_chunk<I, T>(
        &self,
        direction: &Direction,
        reference: &mut [T],
    ) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T>,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum;
}

pub trait ComputeMinMax {
    fn min_max_whole<T>(&self, direction: &Direction) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: NumCast + Copy + PartialOrd + NumericOps;

    fn min_max_chunk<T>(
        &self,
        direction: &Direction,
        reference: (&mut Vec<T>, &mut Vec<T>),
    ) -> anyhow::Result<()>
    where
        T: NumCast + Copy + PartialOrd + NumericOps;
}
