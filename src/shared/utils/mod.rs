use anndata::data::SelectInfoElem;
use ndarray::Slice;
use polars::prelude::{CsvParseOptions, CsvReadOptions, DataFrame, NullValues, SerReader};

pub(crate) fn select_info_elem_to_indices(
    elem: &SelectInfoElem,
    bound: usize,
) -> anyhow::Result<Vec<usize>> {
    match elem {
        SelectInfoElem::Index(indices) => {
            // For Index, we just need to verify that all indices are within bounds
            for &idx in indices {
                if idx >= bound {
                    anyhow::bail!("Index out of bounds: {} >= {}", idx, bound);
                }
            }
            Ok(indices.clone())
        }
        SelectInfoElem::Slice(slice) => {
            let Slice { start, end, step } = *slice;
            let end = end.unwrap_or(bound as isize);

            // Ensure the slice is within bounds
            if start as usize >= bound || end as usize > bound {
                anyhow::bail!(
                    "Slice out of bounds: start={}, end={}, bound={}",
                    start,
                    end,
                    bound
                );
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

pub(crate) fn dataframe_from_csv_bytes(
    bytes: &[u8],
    separator: u8,
    has_header: bool,
    infer_schema_length: Option<usize>,
) -> anyhow::Result<DataFrame> {
    let cursor = std::io::Cursor::new(bytes);
    let csv_parse_options = CsvParseOptions::default()
        .with_separator(separator)
        .with_quote_char(Some(b'"'))
        .with_null_values(Some(NullValues::AllColumnsSingle("NA".to_string().into())));

    let df = CsvReadOptions::default()
        .with_has_header(has_header)
        .with_infer_schema_length(infer_schema_length)
        .with_parse_options(csv_parse_options)
        .into_reader_with_file_handle(cursor)
        .finish()?;

    Ok(df)
}
