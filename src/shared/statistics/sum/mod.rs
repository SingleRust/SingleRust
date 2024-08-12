use anndata::{ArrayData, ArrayElemOp};

use crate::shared::Direction;

/// computes the non_zero values in wither column or row direction for a CSC
pub fn whole(arrayd: ArrayData, direction: Direction) -> anyhow::Result<Vec<f64>> {
    match arrayd {
        ArrayData::Array(array) => todo!("Not implemented yet!"),
        ArrayData::CsrMatrix(csr) => super::helper::csr::sum_whole(csr, direction),
        ArrayData::CsrNonCanonical(csc) => todo!("Not implemented yet!"),
        ArrayData::CscMatrix(csc) => super::helper::csc::sum_whole(csc, direction),
        ArrayData::DataFrame(df) => todo!("Not implemented yet!"),
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
                super::helper::csc::sum_chunk(csc, &direction, &mut return_vec)?
            }
            ArrayData::CsrMatrix(csr) => {
                super::helper::csr::sum_chunk(csr, &direction, &mut return_vec)?
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
