use crate::memory::processing::dimred::FeatureSelectionMethod;
use crate::memory::utils::{arr1_conversion, arr2_conversion};
use anndata::data::DynCsrMatrix;
use anndata::ArrayData;
use anndata_memory::IMArrayElement;
use anyhow::anyhow;
use ndarray::{Array1, Array2};
use rand::distr::Distribution;
use rand::distr::Uniform;
use rand::rng;
use single_algebra::dimred::pca::{MaskedSparsePCABuilder, SvdFloat};
use single_utilities::traits::FloatOpsTS;
use std::fmt::Debug;
use std::ops::Deref;

pub struct PCAResult<T>
where
    T: FloatOpsTS,
{
    transformed: Array2<T>,
    explained_variance_ratio: Array1<T>,
    cumulative_explained_variance_ratio: Array1<T>,
    feature_importance: Array2<T>,
}
pub fn run_pca_sparse_masked<T>(
    matrix: &IMArrayElement,
    feature_selection_method: Option<FeatureSelectionMethod>,
    center: Option<bool>,
    verbose: Option<bool>,
    n_components: Option<usize>,
    alpha: Option<f64>,
    random_seed: Option<u32>,
    max_iter: Option<usize>,
) -> anyhow::Result<PCAResult<T>>
where
    T: FloatOpsTS + SvdFloat,
{
    let feature_selection_method =
        feature_selection_method.unwrap_or(FeatureSelectionMethod::RandomSelection(1000));
    let shape = matrix.get_shape()?;
    let ncols = shape[1];
    let center = center.unwrap_or(false);
    let verbose = verbose.unwrap_or(false);
    let n_components = n_components.unwrap_or(50);
    let random_seed = random_seed.unwrap_or(42);
    let selected = match feature_selection_method {
        FeatureSelectionMethod::FullFeatures => {
            vec![true; ncols]
        }
        FeatureSelectionMethod::HighlyVariableSelection(vec) => vec,
        FeatureSelectionMethod::RandomSelection(num_genes) => {
            generate_random_mask(ncols, num_genes)
        }
    };
    let read_guard = matrix.0.read_inner();
    let data = read_guard.deref();
    match data {
        ArrayData::CsrMatrix(dyn_csr) => {
            match dyn_csr {
                DynCsrMatrix::F32(csr) => {
                    let mut masked_pca = MaskedSparsePCABuilder::new()
                        .mask(selected)
                        .center(center)
                        .verbose(verbose)
                        .alpha(alpha.unwrap_or(1.0) as f32)
                        .n_components(n_components)
                        .random_seed(random_seed)
                        .build();
                    masked_pca.fit(csr)?;
                    let transformed = masked_pca.transform(csr)?;
                    let explained_variance_ratio = masked_pca.explained_variance_ratio()?;
                    let cumulative_explained_variance_ratio = masked_pca.cumulative_explained_variance_ratio()?;
                    let feature_importance = masked_pca.feature_importances()?;

                    let transformed: Array2<T> = arr2_conversion(transformed)?;
                    let explained_variance_ratio: Array1<T> = arr1_conversion(explained_variance_ratio)?;
                    let cumulative_explained_variance_ratio: Array1<T> = arr1_conversion(cumulative_explained_variance_ratio)?;
                    let feature_importance: Array2<T> = arr2_conversion(feature_importance)?;
                    let res = PCAResult {
                        transformed,
                        explained_variance_ratio,
                        cumulative_explained_variance_ratio,
                        feature_importance,
                    };
                    Ok(res)
                }
                DynCsrMatrix::F64(csr) => {
                    let mut masked_pca = MaskedSparsePCABuilder::new()
                        .mask(selected)
                        .center(center)
                        .verbose(verbose)
                        .alpha(alpha.unwrap_or(1.0) as f64)
                        .n_components(n_components)
                        .random_seed(random_seed)
                        .build();
                    masked_pca.fit(csr)?;
                    let transformed = masked_pca.transform(csr)?;
                    let explained_variance_ratio = masked_pca.explained_variance_ratio()?;
                    let cumulative_explained_variance_ratio = masked_pca.cumulative_explained_variance_ratio()?;
                    let feature_importance = masked_pca.feature_importances()?;

                    let transformed: Array2<T> = arr2_conversion(transformed)?;
                    let explained_variance_ratio: Array1<T> = arr1_conversion(explained_variance_ratio)?;
                    let cumulative_explained_variance_ratio: Array1<T> = arr1_conversion(cumulative_explained_variance_ratio)?;
                    let feature_importance: Array2<T> = arr2_conversion(feature_importance)?;
                    let res = PCAResult {
                        transformed: transformed,
                        explained_variance_ratio: explained_variance_ratio,
                        cumulative_explained_variance_ratio: cumulative_explained_variance_ratio,
                        feature_importance: feature_importance,
                    };
                    Ok(res)
                }
                _ => Err(anyhow!("This datatype is currently not supported, please convert to F32 or F64 first, before running PCA!"))
            }
        }
        _ => Err(anyhow!("This anndata type is currently not supported. Only CSR matrices are supported for now!"))
    }
}

fn generate_random_mask(n_genes: usize, num_random_selection: usize) -> Vec<bool> {
    let mut rng = rng();
    let uniform = Uniform::new(0, n_genes).unwrap();
    let mut vec = vec![false; num_random_selection];
    for _ in 0..num_random_selection {
        let v = uniform.sample(&mut rng);
        vec[v] = true;
    }
    vec
}
