use std::ops::AddAssign;

use num_traits::{Float, NumCast, PrimInt, Unsigned, Zero};
use single_algebra::{Direction, NumericOps};

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

pub(crate) mod number {
    use anndata::{ArrayData, ArrayElemOp};

    use crate::shared::Direction;

    /// computes the non_zero values in wither column or row direction for a CSC
    pub fn whole(arrayd: &ArrayData, direction: Direction) -> anyhow::Result<Vec<u32>> {
        match arrayd {
            ArrayData::Array(_) => todo!("Not implemented yet!"),
            ArrayData::CsrMatrix(csr) => super::helper::csr::number_whole(csr, direction),
            ArrayData::CsrNonCanonical(_) => todo!("Not implemented yet!"),
            ArrayData::CscMatrix(csc) => super::helper::csc::number_whole(csc, direction),
            ArrayData::DataFrame(_) => todo!("Not implemented yet!"),
        }
    }

    pub fn chunked<T: ArrayElemOp>(
        x: &T,
        chunk_size: usize,
        direction: Direction,
        length: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let mut return_vec: Vec<u32> = vec![0; length];
        for (chunk, _, _) in x.iter::<ArrayData>(chunk_size) {
            match chunk {
                ArrayData::CscMatrix(csc) => {
                    super::helper::csc::number_chunk(&csc, &direction, &mut return_vec)?
                }
                ArrayData::CsrMatrix(csr) => {
                    super::helper::csr::number_chunk(&csr, &direction, &mut return_vec)?
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported matrix type, please check again!"
                    ))
                }
            };
        }

        Ok(return_vec)
    }
}
pub(crate) mod sum {
    use anndata::{ArrayData, ArrayElemOp};

    use crate::shared::Direction;

    /// computes the non_zero values in wither column or row direction for a CSC
    pub fn whole(arrayd: &ArrayData, direction: Direction) -> anyhow::Result<Vec<f64>> {
        match arrayd {
            ArrayData::Array(_array) => todo!("Not implemented yet!"),
            ArrayData::CsrMatrix(csr) => super::helper::csr::sum_whole(csr, direction),
            ArrayData::CsrNonCanonical(_csc) => todo!("Not implemented yet!"),
            ArrayData::CscMatrix(csc) => super::helper::csc::sum_whole(csc, direction),
            ArrayData::DataFrame(_df) => todo!("Not implemented yet!"),
        }
    }

    pub fn chunked<T: ArrayElemOp>(
        x: &T,
        chunk_size: usize,
        direction: Direction,
        length: usize,
    ) -> anyhow::Result<Vec<f64>> {
        let mut return_vec: Vec<f64> = vec![0f64; length];
        for (chunk, _, _) in x.iter::<ArrayData>(chunk_size) {
            match chunk {
                ArrayData::CscMatrix(csc) => {
                    super::helper::csc::sum_chunk(&csc, &direction, &mut return_vec)?
                }
                ArrayData::CsrMatrix(csr) => {
                    super::helper::csr::sum_chunk(&csr, &direction, &mut return_vec)?
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported matrix type, please check again!"
                    ))
                }
            };
        }

        Ok(return_vec)
    }
}

pub(crate) mod variance {
    use anndata::ArrayData;

    use crate::Direction;

    pub fn whole(arrayd: &ArrayData, direction: Direction) -> anyhow::Result<Vec<f64>> {
        match arrayd {
            ArrayData::CscMatrix(csc) => super::helper::csc::variance_whole(csc, direction),
            ArrayData::CsrMatrix(csr) => super::helper::csr::variance_whole(csr, direction),
            _ => todo!("This is not implemented yet!"),
        }
    }
}

pub(crate) mod minmax {
    use crate::Direction;
    use anndata::ArrayData;

    pub fn whole(arrayd: &ArrayData, direction: Direction) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        match arrayd {
            ArrayData::CscMatrix(csc) => super::helper::csc::min_max_whole(csc, direction),
            ArrayData::CsrMatrix(csr) => super::helper::csr::min_max_whole(csr, direction),
            _ => todo!("This is not implemented yet!"),
        }
    }
}

pub(crate) mod stddev {
    use crate::Direction;
    use anndata::ArrayData;

    pub fn whole(arrayd: &ArrayData, direction: Direction) -> anyhow::Result<Vec<f64>> {
        match arrayd {
            ArrayData::CscMatrix(csc) => super::helper::csc::std_dev_whole(csc, direction),
            ArrayData::CsrMatrix(csr) => super::helper::csr::std_dev_whole(csr, direction),
            _ => todo!("This is not implemented yet!"),
        }
    }
}

pub(crate) mod helper;
