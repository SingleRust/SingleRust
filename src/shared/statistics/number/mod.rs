use anndata::{ArrayData, ArrayElemOp};

use crate::shared::Direction;

/// computes the non_zero values in wither column or row direction for a CSC
pub fn whole(arrayd: ArrayData, direction: Direction) -> anyhow::Result<Vec<u32>> {
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
                super::helper::csc::number_chunk(csc, &direction, &mut return_vec)?
            }
            ArrayData::CsrMatrix(csr) => {
                super::helper::csr::number_chunk(csr, &direction, &mut return_vec)?
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
