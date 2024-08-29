use anndata::data::SelectInfoElem;
use ndarray::{ArrayView, Ix1};

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
        selection[1] = SelectInfoElem::Index(vars_idx);
    }

    Ok(selection)
}
