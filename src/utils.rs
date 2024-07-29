use anndata::{data::SelectInfoElem, AnnData, AnnDataOp, ArrayData, ArrayElemOp, Backend};
use anyhow::anyhow;
use nalgebra_sparse::{CscMatrix, CsrMatrix};

pub fn get_gene_counts<B: Backend>(adata: &AnnData<B>, gene_index: usize) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();

    let shape = x.shape().ok_or_else(|| anyhow!("X matrix shape not found"))?;
    if gene_index >= shape[1] {
        return Err(anyhow!("Gene index out of bounds"));
    }
    let gene_select = SelectInfoElem::Index(vec![gene_index]);
    let gene_counts = x.slice_axis::<ArrayData, _>(1, gene_select)?
    .ok_or_else(|| anyhow!("Failed to slice X matrix"))?;

    match gene_counts {
        ArrayData::CsrMatrix(csr) => {
            let csr: CsrMatrix<f64> = csr.try_into()?;
            csr_column_to_vec(&csr, gene_index)
        },
        ArrayData::CscMatrix(csc) => {
            let csc: CscMatrix<f64> = csc.try_into()?;
            csc_column_to_vec(&csc, 0)  // We use 0 here because the slice is already a single column
        },
        _ => Err(anyhow::anyhow!("Unsupported array type for X"))
    }
}

pub fn csr_column_to_vec(csr: &CsrMatrix<f64>, col_index: usize) -> anyhow::Result<Vec<f64>> {
    let n_rows = csr.nrows();
    let mut dense = vec![0.0; n_rows];

    for (row, col, &value) in csr.triplet_iter() {
        if col == col_index {
            dense[row] = value;
        }
    }

    Ok(dense)
}

pub fn csc_column_to_vec(csc: &CscMatrix<f64>, col_index: usize) -> anyhow::Result<Vec<f64>> {
    let n_rows = csc.nrows();
    let mut dense = vec![0.0; n_rows];

    if col_index < csc.ncols() {
        let start = csc.col_offsets()[col_index];
        let end = csc.col_offsets()[col_index + 1];

        for (idx, &row) in csc.row_indices()[start..end].iter().enumerate() {
            dense[row] = csc.values()[start + idx];
        }
    }

    Ok(dense)
}

pub fn get_cell_counts<B: Backend>(adata: &AnnData<B>, cell_index: usize) -> anyhow::Result<Vec<f64>> {
    let x = adata.x();
    let shape = x.shape().ok_or_else(|| anyhow!("X matrix shape not found"))?;

    if cell_index >= shape[0] {
        return Err(anyhow!("Cell index out of bounds"));
    }

    let cell_select = SelectInfoElem::Index(vec![cell_index]);
    let cell_counts = x.slice_axis::<ArrayData, _>(0, cell_select)?.ok_or_else(|| anyhow!("Failed to slice X matrix"))?;

    match cell_counts {
        ArrayData::CscMatrix(csc) => {
            let csc: CscMatrix<f64> = csc.try_into()?;
            csc_row_to_vec(&csc, 0)
        },
        ArrayData::CsrMatrix(csr) => {
            let csr: CsrMatrix<f64> = csr.try_into()?;
            csr_row_to_vec(&csr, 0)
        },
        ArrayData::CsrNonCanonical(csr_nr) => {
            let csr: CsrMatrix<f64> = csr_nr.canonicalize()
            .map_err(|_| anyhow!("Failed to canonicalize CSRMatrix"))?
            .try_into()?;
            csr_row_to_vec(&csr, 0)
        },
        _ => Err(anyhow!("This datatype is not currently supported!w"))
    }

}

fn csr_row_to_vec(csr: &CsrMatrix<f64>, row_index: usize) -> anyhow::Result<Vec<f64>> {
    let n_cols = csr.ncols();
    let mut dense = vec![0.0; n_cols];
    if row_index < csr.nrows() {
        let start = csr.row_offsets()[row_index];
        let end = csr.row_offsets()[row_index + 1];

        for (idx, &col) in csr.col_indices()[start..end].iter().enumerate() {
            dense[col] = csr.values()[start + idx];
        }
    }
    Ok(dense)
}

fn csc_row_to_vec(csc: &CscMatrix<f64>, row_index: usize) -> anyhow::Result<Vec<f64>> {
    let n_cols = csc.ncols();
    let mut dense = vec![0.0; n_cols];

    for (col, row, &value) in csc.triplet_iter() {
        if row == row_index {
            dense[col] = value;
        }
    }
    Ok(dense)
}


#[cfg(test)]
mod tests {

    use anndata::{AnnData, Backend};
    use anndata_hdf5::H5;
    use std::time::Instant;
    use crate::*;

    fn load_test_data() -> anyhow::Result<AnnData<H5>> {
        let h5data = anndata_hdf5::H5::open("/home/idiks/RESEARCH/SingleRust/data/1m_cellxgene_hsa.h5ad").expect("Unable to open H5 file at specified location!");
        let dataset = anndata::AnnData::<H5>::open(h5data).expect("Unable to open Anndata object from H5 file!");
        Ok(dataset)
    }

    #[test]
    fn test_cell_counts() {
        let adata = load_test_data().expect("Unable to load test data!");
        let now = Instant::now();
        let subset_cell = utils::get_cell_counts(&adata, 0).expect("Could not read one slice of data!");
        let elapsed = now.elapsed();
        println!("Size of the dataset: {}", subset_cell.len());
        println!("It took about {:.2?} seconds", elapsed);

    }
}