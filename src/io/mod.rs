use std::path::Path;

use anndata::{AnnData, Backend};
use anndata_hdf5::H5;

pub fn load_h5ad<P: AsRef<Path>>(ref path: P) -> Result<AnnData<H5>, Box<dyn std::error::Error>> {
    if !file_exists(path) {
        return Err("File does not exist".into());
    }
    let file = H5::open(path)?;
    let anndata = AnnData::open(file)?;
    Ok(anndata)
}


fn file_exists<P: AsRef<Path>>(ref path: P) -> bool {
    Path::new(path.as_ref()).exists()
}

#[cfg(test)]
mod tests {
    use anndata::AnnDataOp;

    use super::*;

    #[test]
    fn test_file_exists() {
        assert_eq!(file_exists("data/sciPlex2_A549_zero_dose.h5ad"), true);
        assert_eq!(file_exists("data/sciPlex2_A549_zero_dose2.h5ad"), false);
    }

    #[test]
    fn test_load_h5ad() {
        let anndata = load_h5ad("data/sciPlex1_HEK293T.h5ad").expect("Failed to load h5ad");
        let var = anndata.var_names().len();
        assert_eq!(var > 100, true);
    }
}
