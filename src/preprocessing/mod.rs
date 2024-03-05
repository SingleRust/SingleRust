use std::error::Error;

use anndata::{AnnData, AnnDataOp, Backend};
use rayon::prelude::*;

pub fn select_genes<B: Backend>(adata: &AnnData<B>, _genes_to_keep: Vec<String>) -> Result<bool, Box<dyn std::error::Error>> {
    let _var = adata.read_var()?;
    //let index = var.index();

    todo!("This feature hasn't been implemented yet, please check GitHub for the most current progress!")
}

pub fn filter_cells_by_count<B: Backend>(_adata: &AnnData<B>, _upper_limit: Option<i32>, _lower_limit: Option<i32>) {
    todo!("This feature hasn't been implemented yet, please check GitHub for the most current progress!")
}

pub fn get_variability_of_genes<B: Backend>(adata: &AnnData<B>) -> Result<Vec<f64>, Box<dyn Error>> {
    let matrix = crate::utils::adata::extract_csc_matrix(adata)?;

    let ncols = matrix.ncols();
    let variances: Vec<f64> = (0..ncols).into_par_iter().map(|col_idx| {
        // Assuming `get_col` returns an Option, we handle it directly.
        matrix.get_col(col_idx).map_or_else(
            || panic!("Failed to get column"), // Handle the None case, could also return an error
            |col| {
                let values = col.values();
                if col.nnz() == 0 {
                    return 0.0;
                }
                let mean = values.iter().sum::<f64>() / col.nnz() as f64;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.nnz() as f64;
                variance // Return the variance for this column
            },
        )
    }).collect();

    Ok(variances)
}


#[cfg(test)]
mod test {
    use anndata::AnnDataOp;

    #[test]
    fn test_select_genes() {
        let path_to_file = std::string::String::from("data/sciPlex1_HEK293T.h5ad");
        let anndata = crate::io::load_h5ad(&path_to_file, None).expect("Failed to load h5ad");
        println!("{:?} Number of variables", anndata.n_vars());
        println!("Calculating variances...");
        let variance = crate::preprocessing::get_variability_of_genes(&anndata).expect("Failed to get gene variance");
        assert_eq!(variance.len(), anndata.n_vars());
    }
}



