use crate::connectors::utils::download_resource;
use crate::shared::utils::dataframe_from_csv_bytes;
use anyhow::anyhow;
use polars::export::arrow::array::ViewType;
use polars::prelude::pivot::pivot;
use polars::prelude::{ChunkCompareEq, DataFrame, PlSmallStr};
use single_utilities::types::PathwayNetwork;
use tokio::runtime::Runtime;

const OMNIPATH_BASE_URL: &str = "https://omnipathdb.org/annotations?databases=";

const LICENSE_TYPES: [&str; 3] = ["academic", "commercial", "nonprofit"];

pub fn load_resource(name: &str, license: &str, tax_id: Option<&str>) -> anyhow::Result<DataFrame> {
    if !LICENSE_TYPES.contains(&license) {
        return Err(anyhow!(
            "This license is not an allowed type, available are: academic, commercial, nonprofit."
        ));
    }

    let path = OMNIPATH_BASE_URL.to_owned() + format!("{}&license={}", name, license).as_str();

    let results = Runtime::new().unwrap().block_on(download_resource(&path))?;
    let res_bytes = results.to_bytes();
    let df = dataframe_from_csv_bytes(res_bytes, b'\t', true, None)?;
    let df = process_omnipath_dataframe(df, tax_id)?;
    Ok(df)
}

fn process_omnipath_dataframe(
    mut df: DataFrame,
    tax_id: Option<&str>,
) -> anyhow::Result<DataFrame> {
    let df = df.select(["genesymbol", "label", "value", "record_id"])?;

    let mut pivoted = pivot(
        &df,
        ["value"],
        Some(["genesymbol", "record_id"]),
        Some(["label"]),
        false,
        None,
        None,
    )?;

    if let Some(org) = tax_id {
        let p: PlSmallStr = org.into();
        if pivoted.get_column_names().contains(&&p) {
            let mask = pivoted.column("ncbi_tax_id")?.str()?.equal(org);
            pivoted = pivoted.filter(&mask)?;
        }
    }

    let cols_to_remove = ["record_id", "entity_type", "_entity_type"];
    let mut final_df = pivoted;

    for col in cols_to_remove {
        let c: PlSmallStr = col.into();
        if final_df.get_column_names().contains(&&c) {
            final_df = final_df.drop(col)?;
        }
    }
    Ok(final_df)
}

pub fn construct_network_from_panglaodb(
    license: &str,
    tax_id: Option<&str>,
    features: Vec<String>,
    tmin: u32,
) -> anyhow::Result<PathwayNetwork> {
    let df = load_resource("panglaodb", license, tax_id)?;
    let names: Vec<String> = df
        .column("cell_type")?
        .str()?
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_string()))
        .collect();
    let targets: Vec<String> = df
        .column("genesymbol")?
        .str()?
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_string()))
        .collect();
    Ok(PathwayNetwork::new_from_vec(
        names, targets, None, features, tmin,
    ))
}
