use std::{ops::Deref, path::Path};
use log::{log, Level};

use anndata_memory::IMAnnData;

use crate::{shared::plot, PcaPlotSettings};


pub fn plot_pca<P: AsRef<Path>>(anndata: &IMAnnData, colors: Option<&[String]>, output_path: P, settings: &PcaPlotSettings) -> anyhow::Result<()> {

    let obsm = anndata.obsm();
    log!(Level::Debug, "Loading Obsm");
    let data = obsm.get_array_shallow("X_pca").unwrap();
    log!(Level::Debug, "Getting data shallow");
    let read_guard = data.0.read_inner();
    log!(Level::Debug, "Aquireing read lock");
    let array_data = read_guard.deref();
    log!(Level::Debug, "Derefing");
    
    // TODO: Optimize here to let plot pca take an array data value and then use macros in order to implement it for different types
    plot::plot_pca_array_data(array_data, colors, output_path, settings)
}