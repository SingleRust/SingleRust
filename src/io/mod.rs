use std::path::Path;

use anndata::{AnnData, Backend};
use anndata_hdf5::H5;
use anndata_memory::{convert_to_in_memory, IMAnnData};

pub enum FileScope {
    Read = 0,
    ReadWrite = 1,
}

pub fn read_h5ad<P: AsRef<Path>>(path_to_file: P, scope: FileScope, enable_cache: bool) -> anyhow::Result<AnnData<H5>> {
    let h5_file = match scope {
        FileScope::Read => H5::open(path_to_file)?,
        FileScope::ReadWrite => H5::open_rw(path_to_file)?,
    };
    let adata = AnnData::<H5>::open(h5_file)?;
    if enable_cache {adata.get_x().inner().enable_cache();}
    Ok(adata)
}

pub fn read_h5ad_memory<P: AsRef<Path>>(path_to_file: P) -> anyhow::Result<IMAnnData> {
    let adata = read_h5ad(path_to_file, FileScope::Read, false)?;
    convert_to_in_memory(adata)
}