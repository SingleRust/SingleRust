use std::path::Path;

use anndata::{AnnData, Backend};
use anndata_hdf5::H5;

pub enum FileScope {
    Read = 0,
    ReadWrite = 1,
}

pub fn read_h5ad<P: AsRef<Path>>(path_to_file: P, scope: FileScope) -> anyhow::Result<AnnData<H5>> {
    let h5_file = match scope {
        FileScope::Read => H5::open(path_to_file)?,
        FileScope::ReadWrite => H5::open_rw(path_to_file)?,
    };
    let adata = AnnData::<H5>::open(h5_file)?;
    Ok(adata)
}
