pub mod processing;
pub mod utils;

pub use crate::*;

#[cfg(test)]
mod tests {
    use anndata::Backend;
    use anndata_hdf5::H5;
    use std::time::Instant;
    use super::*;

    #[test]
    fn it_works() {
        let now = Instant::now();
        let h5data = anndata_hdf5::H5::open("/home/idiks/RESEARCH/SingleRust/data/1m_cellxgene_hsa.h5ad").expect("Unable to open H5 file at specified location!");
        let dataset = anndata::AnnData::<H5>::open(h5data).expect("Unable to open Anndata object from H5 file!");
        let elapsed = now.elapsed();
        println!("{}", dataset);
        println!("It took about {:.2?} seconds", elapsed);

        let now = Instant::now();
        let subset_gene = utils::get_gene_counts(&dataset, 0).expect("Could not read one slice of data!");
        let elapsed = now.elapsed();
        println!("Size of the dataset: {}", subset_gene.len());
        println!("It took about {:.2?} seconds", elapsed);
    }



}
