use anndata::data::SelectInfoElem;
use ndarray::Slice;

pub(crate) fn select_info_elem_to_indices(elem: &SelectInfoElem, bound: usize) -> anyhow::Result<Vec<usize>> {
    match elem {
        SelectInfoElem::Index(indices) => {
            // For Index, we just need to verify that all indices are within bounds
            for &idx in indices {
                if idx >= bound {
                    anyhow::bail!("Index out of bounds: {} >= {}", idx, bound);
                }
            }
            Ok(indices.clone())
        },
        SelectInfoElem::Slice(slice) => {
            let Slice { start, end, step } = *slice;
            let end = end.unwrap_or(bound as isize);
            
            // Ensure the slice is within bounds
            if start as usize >= bound || end as usize > bound {
                anyhow::bail!("Slice out of bounds: start={}, end={}, bound={}", start, end, bound);
            }

            // Generate indices based on the slice
            let indices: Vec<usize> = (start..end)
                .step_by(step as usize)
                .map(|i| i as usize)
                .collect();

            Ok(indices)
        }
    }
}