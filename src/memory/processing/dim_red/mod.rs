use core::num;
use std::ops::Deref;

use anndata::{data::SelectInfoElem, ArrayData, Data, HasShape};
use anndata_memory::{IMAnnData, IMArrayElement};
use anyhow::Ok;
use log::{log, Level};
use ndarray::s;
use ndarray::{Array2};
use rand::{seq::SliceRandom, thread_rng};
use rayon::ThreadPoolBuilder;
use single_algebra::pca::{PCABuilder, SVDImplementation};

use crate::{memory::statistics, shared::{convert_to_array_f64_selected, FeatureSelection}};

pub enum SVDMode {
    #[cfg(feature="faer")]
    FAERRSVD,
    #[cfg(feature="lapack")]
    LAPACK
}


pub fn pca_inplace<S: SVDImplementation>(anndata: &mut IMAnnData, n_components: Option<usize>, center: Option<bool>, scale: Option<bool>, n_threads: Option<usize>, feature_selection: &FeatureSelection, svd_mode: S) -> anyhow::Result<()> {
    log!(Level::Debug, "------------- Calculatung PCA inplace -------------");
    let x_data = anndata.x();
    log!(Level::Debug, "Getting X data from the anndata object");
    let selected_features = select_features(anndata, feature_selection)?;
    let select_full = SelectInfoElem::full();
    let selection_col = SelectInfoElem::Index(selected_features.clone());
    let read_guard = x_data.0.read_inner();
    log!(Level::Debug, "Aquired read guard from the IMArrayElement");
    let array_data = read_guard.deref();
    let dense = convert_to_array_f64_selected(array_data, array_data.shape(), &select_full, &selection_col)?;
    log!(Level::Debug, "Converted to dense representation of sparse data");
    drop(read_guard);
    log!(Level::Debug, "Dropped read guard");
    let x_column = dense.column(0);
    let y_column = dense.column(1);
    log!(Level::Debug, "First few x values: {:?}", &x_column.slice(s![..5]));
    log!(Level::Debug, "First few y values: {:?}", &y_column.slice(s![..5]));
    for n in 0..dense.ncols() {
        let col_n = dense.column(n);
        if col_n.iter().any(|&x| x > 0.0_f64) {
            log!(Level::Debug, "Found one bigger than 0 in col: {}", n);
        }
        if col_n.iter().all(|&x| x == 0.0_f64) {
            log!(Level::Debug, "All just 0 in col: {}", n);
        }
    }

    let n_components = n_components.unwrap_or(2).min(selected_features.len());
    let mut pca = PCABuilder::new(svd_mode)
        .n_components(n_components)
        .center(center.unwrap_or(true))
        .scale(scale.unwrap_or(true))
        .build();

    log!(Level::Debug, "Created PCA builder for further calculations");

    let thread_pool = ThreadPoolBuilder::new().num_threads(n_threads.unwrap_or(rayon::current_num_threads())).build()?;
    log!(Level::Debug, "Crated threadpool for distibuted calculations");
    log!(Level::Debug, "Current number of threads: {}", thread_pool.current_num_threads());
    log!(Level::Debug, "{:?}", thread_pool);
    log!(Level::Debug, "Fitting PCA builder to the data");
    thread_pool.install(|| pca.fit(dense.view()))?;
    log!(Level::Debug, "Fitted PCA to the data!");
    log!(Level::Debug, "Transforming the data....");
    let transformed = thread_pool.install(|| pca.transform(dense.view()))?;

    let x_column = transformed.column(0);
    let y_column = transformed.column(1);
    log!(Level::Debug, "First few x values: {:?}", &x_column.slice(s![..5]));
    log!(Level::Debug, "First few y values: {:?}", &y_column.slice(s![..5]));
    log!(Level::Debug, "Transformed data succcessfully using multiple threads.");

    //let loadings = pca.compute_loadings();
    let explained_variance_ratio = pca.explained_variance_ratio().map(|arr| arr.to_vec());

    attach_pca_results(
        anndata,
        transformed,
        //loadings,
        None,
        explained_variance_ratio,
        selected_features,
        n_components
    )?;

    log!(Level::Debug, "PCA results attached to AnnData object");

    Ok(())

}

fn attach_pca_results(
    anndata: &mut IMAnnData,
    transformed: Array2<f64>,
    loadings: Option<Array2<f64>>,
    explained_variance_ratio: Option<Vec<f64>>,
    selected_features: Vec<usize>,
    n_components: usize
) -> anyhow::Result<()> {
    // Attach PCA transformed data to obsm
    let obsm = anndata.obsm();
    obsm.add_array("X_pca".to_string(), IMArrayElement::new(ArrayData::from(transformed)))?;

    // Attach loadings to varm if available
    if let Some(loadings) = loadings {
        let varm = anndata.varm();
        let mut full_loadings = Array2::zeros((anndata.n_vars(), n_components));
        for (i, &feature_idx) in selected_features.iter().enumerate() {
            if i < loadings.nrows() {
                full_loadings.row_mut(feature_idx).assign(&loadings.row(i));
            }
        }
        varm.add_array("PCA_loadings".to_string(), IMArrayElement::new(ArrayData::from(full_loadings)))?;
    }

    Ok(())
}

fn select_features(anndata: &IMAnnData, feature_selection: &FeatureSelection) -> anyhow::Result<Vec<usize>> {
    match feature_selection {
        FeatureSelection::HighlyVariableCol(col) => {
            let var_df = anndata.var().get_data();

            let bool_mask = var_df.column(col)
            .map_err(|e| anyhow::anyhow!("Error accessing column '{}' : {}", col, e))?
            .bool()
            .map_err(|e| anyhow::anyhow!("Column '{}' is not boolean: {}", col, e))?;

            Ok(bool_mask.iter().enumerate().filter_map(|(i,v)| if v.unwrap_or(false) { Some(i) } else { None }).collect())
        },
        FeatureSelection::HighlyVariable(num_genes) => {
            let variances = statistics::compute_variance(anndata, crate::Direction::Column)?;
            let mut index_variances: Vec<(usize, f64)> = variances.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            index_variances.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            Ok(index_variances.iter().take(*num_genes).map(|&(i, _)| i).collect())
        },
        FeatureSelection::Randomized(num_genes) => {
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..anndata.n_vars()).collect();
            indices.shuffle(&mut rng);
            Ok(indices.into_iter().take(*num_genes).collect())
        },
        FeatureSelection::VarianceThreshold(threshold) => {
            let variances = statistics::compute_variance(anndata, crate::Direction::Column)?;
            Ok(variances.iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > *threshold { Some(i) } else { None })
                .collect())
        },
        FeatureSelection::None => Ok((0..anndata.n_vars()).collect()),
    }
}


// Calculate TSNE representation of the data.
// pub fn tsne(
//     adata: &IMAnnData,
//     perplexity: Option<f64>,
//     approx_theshold: Option<f64>,
// ) -> anyhow::Result<Array2<f64>> {
//     let x_data = adata.x();
//     let read_guard = x_data.0.read_inner();
//     let x = read_guard.deref();
//     let dense_array = convert_to_array_f64(x)?;
//     let labels = Array2::from_shape_fn((adata.n_obs(), 1), |(i, _)| i as u32);
//     let dataset = Dataset::new(dense_array, labels).with_feature_names(
//         (0..adata.n_vars())
//             .map(|i| format!("feature_{}", i))
//             .collect(),
//     );

//     let pca: Pca<f64> = Pca::params(3).whiten(true).fit(&dataset)?;
//     let transformed = pca.transform(dataset);
//     let transformed = TSneParams::embedding_size(2)
//         .perplexity(perplexity.unwrap_or(10.0))
//         .approx_threshold(approx_theshold.unwrap_or(0.1))
//         .transform(transformed)?;
//     let tsne_coords: Vec<Vec<f64>> = transformed
//         .sample_iter()
//         .map(|(x, _)| vec![x[0], x[1]])
//         .collect();
//     let tsne_array = Array2::from_shape_vec(
//         (tsne_coords.len(), 2),
//         tsne_coords.into_iter().flatten().collect(),
//     )?;
//     Ok(tsne_array)
// }

// /// Calculate TSNE representation of the data and store it in the AnnData object, intercompatible with the python version.
// pub fn tsne_inplace(
//     adata: &mut IMAnnData,
//     perplexity: Option<f64>,
//     approx_theshold: Option<f64>,
// ) -> anyhow::Result<()> {
//     let x_data = adata.x();
//     let read_guard = x_data.0.read_inner();
//     let x = read_guard.deref();
//     let dense_array = convert_to_array_f64(x)?;
//     let labels = Array2::from_shape_fn((adata.n_obs(), 1), |(i, _)| i as u32);
//     let dataset = Dataset::new(dense_array, labels).with_feature_names(
//         (0..adata.n_vars())
//             .map(|i| format!("feature_{}", i))
//             .collect(),
//     );

//     let pca: Pca<f64> = Pca::params(3).whiten(true).fit(&dataset)?;
//     let transformed = pca.transform(dataset);
//     let transformed = TSneParams::embedding_size(2)
//         .perplexity(perplexity.unwrap_or(10.0))
//         .approx_threshold(approx_theshold.unwrap_or(0.1))
//         .transform(transformed)?;
//     let tsne_coords: Vec<Vec<f64>> = transformed
//         .sample_iter()
//         .map(|(x, _)| vec![x[0], x[1]])
//         .collect();
//     let tsne_array = Array2::from_shape_vec(
//         (tsne_coords.len(), 2),
//         tsne_coords.into_iter().flatten().collect(),
//     )?;
    
//     let obsm = adata.obsm();
//     obsm.add_array("X_tsne".to_string(), IMArrayElement::new(ArrayData::from(tsne_array)))
// }
