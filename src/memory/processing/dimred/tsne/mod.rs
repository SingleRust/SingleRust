use anndata_memory::IMAnnData;
use anyhow::anyhow;
use ndarray::ArrayD;
use single_utilities::traits::FloatOpsTS;

pub fn run<T: FloatOpsTS>(
    adata: IMAnnData,
    output_dim: u8,
    perplexity: f32,
    epochs: usize,
    theta: f32,
    pp_key: Option<String>
) -> anyhow::Result<ArrayD<T>> {
    let obsm_keys = adata.obsm().keys();
    let pp_key = pp_key.unwrap_or("X_pca".to_string());
    if !obsm_keys.contains(&pp_key) {
        return Err(anyhow!("X_pca not found in obsm_keys"));
    }



    

}

