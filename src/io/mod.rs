use std::path::Path;

use anndata::{AnnData, Backend};
use anndata_hdf5::{H5, H5File};
use console::style;

use crate::utils::matrix_market::MatrixMarketReader;

pub enum OpenScope {
    Read = 0,
    ReadWrite = 1,
}

///
///
/// # Arguments
///
/// * `path`: Path to the H5AD file to be loaded
/// * `scope`: Scope to open the H5AD file, either read or read-write, when no scope is provided, the file is opened in read mode.
///
/// returns: Result<AnnData<H5>, Box<dyn Error, Global>>
///
/// # Examples
///
/// ```
/// let path_to_your_h5ad = String::from("PATH_TO_YOUR_FILE");
/// let anndata = singleRust::io::load_h5ad(&path_to_your_h5ad, Some(singleRust::io::OpenScope::Read)).expect("Failed to load h5ad");
/// ```
pub fn load_h5ad<P: AsRef<Path>>(path: &P, scope: Option<OpenScope>) -> Result<AnnData<H5>, Box<dyn std::error::Error>> {
    if !crate::utils::io::file_exists(path) {
        return Err("File does not exist".into());
    }
    let file: H5File = match scope {
        Some(OpenScope::Read) => H5::open(path)?,
        Some(OpenScope::ReadWrite) => H5::open_rw(path)?,
        None => H5::open(path)?,
    };
    let anndata = AnnData::<H5>::open(file)?;
    Ok(anndata)
}

///
///
/// # Arguments
///
/// * `path_to_csv`: Path to the csv file to be loaded, column names are expected to be in the first row. One column is one cell, one row is one gene.
/// * `path_to_h5`: Optional path to save the AnnData object to a H5AD file, otherwise this is an in-memory object.
///
/// returns: Result<AnnData<H5>, Box<dyn Error, Global>>
///
/// Function that loads a csv file into an AnnData object, optionally saving it to a H5AD file.
///
/// # Examples
///
/// ```
///
/// ```
pub fn load_csv<P: AsRef<Path>>(_path_to_csv: &P, _path_to_h5: Option<&P>) -> Result<AnnData<H5>, Box<dyn std::error::Error>> {
    todo!("This feature hasn't been implemented yet, please check GitHub for the most current progress!")
}

/// Function that takes a mtx file (counts), a features file (genes), a barcodes file (cells) and a path to save the AnnData object to a H5AD file.
///
/// # Arguments
///
/// * `path_to_mtx`: Path to the (possibly compressed) mtx file (counts)
/// * `path_to_features`: Path to the (possibly compressed) features file (genes)
/// * `path_to_barcodes`: Path to the (possibly compressed) barcodes file (cells)
/// * `path_to_h5`: path to save the AnnData object to a H5AD file
/// * `sorted`: Whether the mtx file is sorted or not
/// * 
///
/// returns: Result<AnnData<H5>, Box<dyn Error, Global>>
///
/// # Examples .
///
/// ```
/// let basepath = "data/mtx";
/// let matrix_path = format!("{}/matrix.mtx.gz", basepath);
/// let barcodes_path = format!("{}/barcodes.tsv.gz", basepath);
/// let features_path = format!("{}/features.tsv.gz", basepath);
/// let h5_path = format!("{}/test.h5ad", basepath);
/// let anndata = singleRust::io::load_mtx(&matrix_path, Some(&features_path), Some(&barcodes_path), &h5_path, Some(false), Some(true), Some(true)).expect("Failed to load mtx");
/// ```
pub fn load_mtx<P: AsRef<Path>>(path_to_mtx: &P, path_to_features: Option<&P>, path_to_barcodes: Option<&P>, path_to_h5: &P, sorted: Option<bool>, load_transposed: Option<bool>, show_progress: Option<bool>) -> Result<AnnData<H5>, Box<dyn std::error::Error>> {
    // Create a new progress bar
    let show_progress = show_progress.unwrap_or(false);

    // Check if all files exist before proceeding
    if !crate::utils::io::file_exists(path_to_mtx) {
        return Err("Matrix-File does not exist".into());
    }
    if show_progress {
        println!("{} {} Checking if files exist...",
            style("[1/7]").bold().dim(),
            crate::utils::emoji::LOOKING_GLASS
        );
    }

    if let Some(path) = path_to_features {
        if !crate::utils::io::file_exists(path) {
            return Err("Features-File does not exist".into());
        }
    }
    

    if let Some(path) = path_to_barcodes {
        if !crate::utils::io::file_exists(path) {
            return Err("Barcodes-File does not exist".into());
        }
    }

    let mut reader = MatrixMarketReader::from_path(path_to_mtx)?;
    if show_progress {
        println!("{} {} Initializing Matrix Market reader!",
            style("[2/7]").bold().dim(),
            crate::utils::emoji::GEAR
        );
    }

    if let Some(sorted) = sorted {
        reader = reader.set_sorted(sorted);
    }

    if let Some(path) = path_to_features {
        reader = reader.read_var_names(path)?;
    }
    
    if show_progress {
        println!("{} {} Reading var names...",
            style("[3/7]").bold().dim(),
            crate::utils::emoji::FILE
        );
    }
    
    if let Some(path) = path_to_barcodes {
        reader = reader.read_obs_names(path)?;
    }
    
    if show_progress {
        println!("{} {} Reading obs names...",
            style("[4/7]").bold().dim(),
            crate::utils::emoji::FILE
        );
    }

    let data = AnnData::<H5>::new(path_to_h5)?;
    
    if show_progress {
        println!("{} {} Creating AnnData object...",
            style("[5/7]").bold().dim(),
            crate::utils::emoji::FOLDER
        );
    }

    reader.finish(&data, load_transposed.unwrap_or(false))?;
    
    if show_progress {
        println!("{} {} Finishing up...",
            style("[6/7]").bold().dim(),
            crate::utils::emoji::FLOPPY
        );
    }

    if show_progress {
        println!("{} {} Done!",
            style("[7/7]").bold().dim(),
            crate::utils::emoji::CHECKMARK
        );
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use anndata::AnnDataOp;

    use super::*;

    #[test]
    fn test_load_h5ad() {
        let path_to_file = String::from("data/sciPlex1_HEK293T.h5ad");
        let anndata = load_h5ad(&path_to_file, None).expect("Failed to load h5ad");
        let var = anndata.var_names().len();
        assert_eq!(var > 100, true);
    }

    #[test]
    fn test_load_mtx() {
        let basepath = "data/mtx";
        let matrix_path = format!("{}/matrix.mtx.gz", basepath);
        let barcodes_path = format!("{}/barcodes.tsv.gz", basepath);
        let features_path = format!("{}/features.tsv.gz", basepath);
        let h5_path = format!("{}/test.h5ad", basepath);
        let anndata = load_mtx(&matrix_path, Some(&features_path), Some(&barcodes_path), &h5_path, Some(false), Some(true), Some(true)).expect("Failed to load mtx");
        assert_eq!(anndata.var_names().len() > 100, true);
    }
}
