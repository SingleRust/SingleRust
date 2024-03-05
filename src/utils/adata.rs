use std::error::Error;
use std::path::Path;

use anndata::{AnnData, AnnDataOp, Backend};
use anndata::data::SelectInfoElem;

pub fn apply_obs_names_to_adata<P: AsRef<Path>, B: Backend>(path: &P, adata: &AnnData<B>) -> Result<(), Box<dyn Error>> {
    let obs_names = crate::utils::io::read_list_to_dataframe_index(path);

    adata.set_obs_names(obs_names?)?;

    Ok(())
}

pub fn apply_var_names_to_adata<P: AsRef<Path>, B: Backend>(path: &P, adata: &AnnData<B>) -> Result<(), Box<dyn Error>> {
    let var_names = crate::utils::io::read_list_to_dataframe_index(path);

    adata.set_var_names(var_names?)?;
    Ok(())
}

pub fn select_by_var_bool_vec<B: Backend>(adata: &AnnData<B>, select_var: &Vec<bool>) -> Result<(), Box<dyn Error>> {
    if select_var.len() != adata.n_vars() {
        return Err("Length of select_vec does not match number of variables in AnnData".into());
    }

    let true_pos_vec = bool_to_usize_vec(select_var);
    let obs_selection = SelectInfoElem::full();
    let var_selection = SelectInfoElem::from(true_pos_vec);
    adata.subset([obs_selection, var_selection])?;
    Ok(())
}

pub fn select_by_obs_bool_vec<B: Backend>(adata: &AnnData<B>, select_obs: &Vec<bool>) -> Result<(), Box<dyn Error>> {
    if select_obs.len() != adata.n_obs() {
        panic!("Length of select_obs does not match number of observations in AnnData");
    }

    let true_pos_vec = bool_to_usize_vec(select_obs);
    let obs_selection = SelectInfoElem::from(true_pos_vec);
    let var_selection = SelectInfoElem::full();
    adata.subset([obs_selection, var_selection])?;
    Ok(())
}

fn bool_to_usize_vec(bool_vec: &[bool]) -> Vec<usize> {
    bool_vec.iter().enumerate().filter(|(_, &x)| x).map(|(i, _)| i).collect::<Vec<usize>>()
}


#[cfg(test)]
mod tests {
    use crate::utils::io::file_exists;

    use super::*;

    #[test]
    fn apply_var_names_test() {
        let matrix_path = String::from("data/mtx/matrix.mtx.gz");
        let h5ad_path = String::from("data/apply_matrix_var_names.h5ad");
        if file_exists(&h5ad_path) {
            std::fs::remove_file(&h5ad_path).expect("Failed to remove file");
        }
        let matrix_adata = crate::io::load_mtx(&matrix_path, None, None, &h5ad_path, None, Some(true), Some(true)).expect("Failed to load mtx");
        let var_names_path = String::from("data/mtx/features.tsv.gz");
        apply_var_names_to_adata(&var_names_path, &matrix_adata).expect("Failed to apply var names");
        assert!(matrix_adata.var_names().len() > 100);
    }

    #[test]
    fn test_subset_bool() {
        let matrix_path = String::from("data/mtx/matrix.mtx.gz");
        let h5ad_path = String::from("data/apply_matrix_subset.h5ad");
        if file_exists(&h5ad_path) {
            std::fs::remove_file(&h5ad_path).expect("Failed to remove file");
        }
        let matrix_adata = crate::io::load_mtx(&matrix_path, None, None, &h5ad_path, None, Some(true), Some(true)).expect("Failed to load mtx");
        let select_vec = (0..matrix_adata.n_vars()).map(|x| x % 2 == 0).collect::<Vec<bool>>();
        select_by_var_bool_vec(&matrix_adata, &select_vec).expect("Failed to subset by bool vec");
        println!("{:?}", matrix_adata.n_vars());
        println!("{:?}", select_vec.iter().filter(|&x| *x).count());
        assert_eq!(matrix_adata.n_vars(), select_vec.iter().filter(|&x| *x).count());
    }
}
