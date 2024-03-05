use anndata::{AnnData, AnnDataOp, Backend};

pub fn select_genes<B: Backend>(adata: &AnnData<B>, _genes_to_keep: Vec<String>) -> Result<bool, Box<dyn std::error::Error>> {
    let _var = adata.read_var()?;
    //let index = var.index();

    todo!("This feature hasn't been implemented yet, please check GitHub for the most current progress!")
}

pub fn filter_cells_by_count<B: Backend>(_adata: &AnnData<B>, _upper_limit: Option<i32>, _lower_limit: Option<i32>) {
    todo!("This feature hasn't been implemented yet, please check GitHub for the most current progress!")
}

pub fn get_variability_of_genes<B: Backend>(adata: &AnnData<B>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    todo!("This feature hasn't been implemented yet, please check GitHub for the most current progress!")
    
    //let gene_variance: Vec<f64> = vec![0.0; adata.n_vars()];
    //let _matrix = adata.get_x().inner();

    //Ok(gene_variance)
    
}




