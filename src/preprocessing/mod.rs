use std::ops::Index;
use anndata::{AnnData, AnnDataOp, Backend};

pub fn select_genes<B: Backend>(adata: &AnnData<B>, genes_to_keep: Vec<String>) -> Result<bool, Box<dyn std::error::Error>> {
    let var = adata.read_var()?;
    //let index = var.index();
    
    todo!()
}

pub fn filter_cells_by_count<B: Backend>(adata: &AnnData<B>, upper_limit: Option<i32>, lower_limit: Option<i32>) {
    todo!()
}




