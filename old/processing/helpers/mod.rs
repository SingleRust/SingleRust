use std::path::Path;

use anndata::{data::SelectInfoElem, AnnData, Backend};
use anndata_hdf5::H5;
use ndarray::{ArrayView, Ix1};

use crate::io::FileScope;

pub fn filter_anndata_inplace<B: Backend>(
    adata: &AnnData<B>,
    obs_mask: Option<ArrayView<'_, bool, Ix1>>,
    var_mask: Option<ArrayView<'_, bool, Ix1>>,
) -> anyhow::Result<()> {
    // Convert masks to indices
    let obs_indices = obs_mask.map(|mask| mask.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect::<Vec<_>>());
    let var_indices = var_mask.map(|mask| mask.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect::<Vec<_>>());

    // Create a selection based on the masks
    let mut selection = vec![SelectInfoElem::full(), SelectInfoElem::full()];
    if let Some(obs_idx) = obs_indices {
        selection[0] = SelectInfoElem::Index(obs_idx);
    }
    if let Some(var_idx) = var_indices {
        selection[1] = SelectInfoElem::Index(var_idx);
    }

    // Apply the selection to the AnnData object
    adata.subset(selection.as_slice())?;

    Ok(())
}

pub fn filter_anndata<B: Backend, P: AsRef<Path>>(
    adata: &AnnData<B>,
    obs_mask: Option<ArrayView<'_, bool, Ix1>>,
    var_mask: Option<ArrayView<'_, bool, Ix1>>,
    file_path: P,
    open_scope: FileScope
) -> anyhow::Result<AnnData<H5>> {
    // Convert masks to indices
    let obs_indices = obs_mask.map(|mask| mask.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect::<Vec<_>>());
    let var_indices = var_mask.map(|mask| mask.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect::<Vec<_>>());

    // Create a selection based on the masks
    let mut selection = vec![SelectInfoElem::full(), SelectInfoElem::full()];
    if let Some(obs_idx) = obs_indices {
        selection[0] = SelectInfoElem::Index(obs_idx);
    }
    if let Some(var_idx) = var_indices {
        selection[1] = SelectInfoElem::Index(var_idx);
    }

    // Apply the selection to the AnnData object
    adata.write_select::<H5, _, _>(selection.as_slice(), file_path.as_ref())?;
    
    crate::io::read_h5ad(file_path, open_scope, true)  
}