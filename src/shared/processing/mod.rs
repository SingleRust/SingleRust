use anndata::data::SelectInfoElem;
use log::{log, Level};
use ndarray::{Array1, Array2, ArrayView, Ix1};
use nshare::{ToNalgebra, ToNdarray2};
use rayon::ThreadPool;

//pub mod pca;

//pub mod pcav2;

pub fn get_select_info_obs(
    obs_mask: Option<ArrayView<'_, bool, Ix1>>,
) -> anyhow::Result<Vec<SelectInfoElem>> {
    // Convert masks to indices
    let obs_indices = obs_mask.map(|mask| {
        mask.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect::<Vec<_>>()
    });

    // Create a selection based on the masks
    let mut selection = vec![SelectInfoElem::full(), SelectInfoElem::full()];
    if let Some(obs_idx) = obs_indices {
        selection[0] = SelectInfoElem::Index(obs_idx);
    }

    Ok(selection)
}

pub fn get_select_info_vars(
    vars_mask: Option<ArrayView<'_, bool, Ix1>>,
) -> anyhow::Result<Vec<SelectInfoElem>> {
    // Convert mask to indices
    let vars_indices = vars_mask.map(|mask| {
        mask.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect::<Vec<_>>()
    });

    // Create a selection based on the mask
    let mut selection = vec![SelectInfoElem::full(), SelectInfoElem::full()];
    if let Some(vars_idx) = vars_indices {
        log!(Level::Debug, "VarIDx length: {}", vars_idx.len());
        selection[1] = SelectInfoElem::Index(vars_idx);
    }

    Ok(selection)
}

pub fn calculate_svd(
    array: ndarray::ArrayView2<f64>,
    thread_pool: &ThreadPool,
) -> anyhow::Result<(Array1<f64>, Array2<f64>)> {
    let m = array.into_nalgebra();
    let svd = thread_pool.install(|| m.svd(false, true));

    // return values:

    let s = Array1::from(svd.singular_values.as_slice().to_vec());
    let vt = svd.v_t.unwrap().into_ndarray2();

    Ok((s, vt))
}
