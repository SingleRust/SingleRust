use std::ops::AddAssign;

use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use single_utilities::traits::NumericOps;
use single_utilities::types::Direction;

pub trait ComputeNonZero {
    fn nonzero_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    fn nonzero_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;

    fn nonzero_whole_masked<T>(
        &self,
        direction: &Direction,
        mask: &[bool],
    ) -> anyhow::Result<Vec<T>>
    where
        T: PrimInt + Unsigned + Zero + AddAssign + Send + Sync;
}

pub trait ComputeSum {
    fn sum_whole<T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_chunk<T>(&self, direction: &Direction, reference: &mut [T]) -> anyhow::Result<()>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn sum_whole_masked<T>(&self, direction: &Direction, mask: &[bool]) -> anyhow::Result<Vec<T>>
    where
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;
}

pub trait ComputeVariance {
    fn variance_whole<I, T>(&self, direction: &Direction) -> anyhow::Result<Vec<T>>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;

    fn variance_chunk<I, T>(
        &self,
        direction: &Direction,
        reference: &mut [T],
    ) -> anyhow::Result<()>
    where
        I: PrimInt + Unsigned + Zero + AddAssign + Into<T> + Send + Sync,
        T: Float + num_traits::NumCast + AddAssign + std::iter::Sum + Send + Sync;
}

pub trait ComputeMinMax {
    fn min_max_whole<T>(&self, direction: &Direction) -> anyhow::Result<(Vec<T>, Vec<T>)>
    where
        T: NumCast + Copy + PartialOrd + NumericOps + Send + Sync;

    fn min_max_chunk<T>(
        &self,
        direction: &Direction,
        reference: (&mut Vec<T>, &mut Vec<T>),
    ) -> anyhow::Result<()>
    where
        T: NumCast + Copy + PartialOrd + NumericOps + Send + Sync;
}

pub trait ComputeNTop {
    fn n_top_whole<T>(&self, direction: &Direction, n: usize) -> anyhow::Result<Vec<T>>
    where
        T: Float + NumCast + Copy + PartialOrd + NumericOps + Send + Sync;
}

pub trait ComputeTopSegmentProportions {
    fn top_segment_proportions(&self, direction: &Direction, ns: &[usize]) -> anyhow::Result<ndarray::Array2<f64>>;
}