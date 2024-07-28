mod processing;
mod utils;


#[cfg(test)]
mod tests {
    use anndata::Backend;
    use anndata_hdf5::H5;
    use std::time::Instant;

    #[test]
    fn it_works() {
        let now = Instant::now();
        let h5data = anndata_hdf5::H5::open("/Users/ian/sideprojects/CRIMELABS-EU/PROJECT-SingleRust/library/data/14k_cellXgene_mmu.h5ad").expect("Unable to open H5 file at specified location!");
        let dataset = anndata::AnnData::<H5>::open(h5data).expect("Unable to open Anndata object from H5 file!");
        let elapsed = now.elapsed();
        println!("{}", dataset);
        println!("It took about {:.2?} seconds", elapsed)
    }

}
