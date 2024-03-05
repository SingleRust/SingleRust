use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anndata::data::DataFrameIndex;
use flate2::read::MultiGzDecoder;

pub fn file_exists<P: AsRef<Path>>(path: &P) -> bool {
    Path::new(path.as_ref()).exists()
}

pub fn is_file_gzipped<P: AsRef<Path>>(path: &P) -> Result<bool, Box<dyn std::error::Error>> {
    if !file_exists(path) {
        return Err("File does not exist".into());
    }

    Ok(MultiGzDecoder::new(File::open(path)?).header().is_some())
}

pub fn open_file_bufread<P: AsRef<Path>>(file: P) -> Result<Box<dyn BufRead>, Box<dyn Error>> {
    fn is_gzipped<P: AsRef<Path>>(file: P) -> Result<bool, Box<dyn Error>> {
        Ok(MultiGzDecoder::new(File::open(file)?).header().is_some())
    }

    let reader: Box<dyn BufRead> = if is_gzipped(&file)? {
        Box::new(BufReader::new(MultiGzDecoder::new(File::open(file)?)))
    } else {
        Box::new(BufReader::new(File::open(file)?))
    };
    Ok(reader)
}

pub fn read_list_to_dataframe_index<P: AsRef<Path>>(path: &P) -> Result<DataFrameIndex, Box<dyn Error>> {
    if !file_exists(path) {
        return Err("File does not exist".into());
    }

    let reader = open_file_bufread(path)?;
    let names: Result<DataFrameIndex, Box<dyn Error>> = reader
        .lines()
        .map(|line| Ok(line?.split('\t').next().unwrap().to_string()))
        .collect();

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_exists() {
        let path_1 = String::from("data/sciPlex2_A549_zero_dose.h5ad");
        let path_2 = String::from("data/sciPlex2_A549_zero_dose2.h5ad");
        assert_eq!(file_exists(&path_1), true);
        assert_eq!(file_exists(&path_2), false);
    }
}