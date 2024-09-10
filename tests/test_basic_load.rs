use std::{ops::Deref, time::Instant};

use log::{log, Level};
use single_algebra::pca::{FaerSVD, LapackSVD};
use single_rust::{
    convert_to_array_f64,
    io::read_h5ad_memory,
    memory::{plot, processing},
    FeatureSelection, FlexValue, PcaPlotSettings,
};

#[test]
fn load_file_test() {
    let in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad",
    )
    .unwrap();
    print!("{}", in_memory_file);
}

#[test]
fn load_file_test_small() {
    let in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad",
    )
    .unwrap();
    print!("{}", in_memory_file);
}

#[test]
fn load_file_test_print_obs() {
    let in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad",
    )
    .unwrap();
    let obs = in_memory_file.obs().get_data();
    println!("{}", obs)
}

#[test]
fn load_file_test_print_var() {
    let in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad",
    )
    .unwrap();
    let var = in_memory_file.var().get_data();
    println!("{}", var)
}

#[test]
fn load_file_test_print_var_names() {
    let in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad",
    )
    .unwrap();
    let var = in_memory_file.var_names();
    println!("{:?}", var)
}

#[test]
fn load_file_with_test() {
    env_logger::init();
    let mut in_memory_file =
        read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad").unwrap();
    let f_sel = FeatureSelection::HighlyVariable(25);
    log!(Level::Debug, "{}", in_memory_file);
    single_rust::memory::processing::dim_red::pca_inplace(
        &mut in_memory_file,
        Some(5),
        None,
        None,
        Some(32),
        &f_sel,
        FaerSVD
    )
    .unwrap();

    log!(Level::Debug, "{}", in_memory_file);
}

#[test]
fn load_file_with_test_plot() {
    env_logger::init();
    let mut in_memory_file = read_h5ad_memory(
        //"/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad",
        "/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad",
    )
    .unwrap();
    let f_sel = FeatureSelection::HighlyVariable(25);
    log!(Level::Debug, "{}", in_memory_file);
    single_rust::memory::processing::dim_red::pca_inplace(
        &mut in_memory_file,
        Some(5),
        None,
        None,
        Some(32),
        &f_sel,
        FaerSVD
    )
    .unwrap();

    log!(Level::Debug, "{}", in_memory_file);

    let pca_settings = PcaPlotSettings::default();

    plot::plot_pca(&in_memory_file, None, "test.png", &pca_settings).unwrap();
}

#[test]
fn load_file_with_test_plot_small_faer() {
    env_logger::init();
    let mut in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad",
        //"/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad"
    )
    .unwrap();
    let n_cells_before = in_memory_file.n_obs();
    log!(Level::Debug, "Performing filtering of all cells");
    single_rust::memory::processing::filter_cells_inplace(
        &mut in_memory_file,
        FlexValue::Absolute(200),
        FlexValue::None,
    )
    .unwrap();
    let n_cells_after = in_memory_file.n_obs();

    log!(
        Level::Debug,
        "Num obs before: {}, Num obs after: {}",
        n_cells_before,
        n_cells_after
    );

    let n_genes_before = in_memory_file.n_vars();
    log!(Level::Debug, "Performing filtering of all genes");
    single_rust::memory::processing::filter_genes_inplace(
        &mut in_memory_file,
        FlexValue::Absolute(3),
        FlexValue::None,
    )
    .unwrap();
    let n_genes_after = in_memory_file.n_vars();

    log!(
        Level::Debug,
        "Num vars before: {}, Num vars after: {}",
        n_genes_before,
        n_genes_after
    );
    let pca_time = Instant::now();
    let f_sel = FeatureSelection::HighlyVariable(25);
    log!(Level::Debug, "{}", in_memory_file);
    single_rust::memory::processing::dim_red::pca_inplace(
        &mut in_memory_file,
        Some(5),
        None,
        Some(true),
        Some(32),
        &f_sel,
        FaerSVD
    )
    .unwrap();
    let duration = pca_time.elapsed();
    log!(Level::Error,"Took time to calculate PCA: {:?}", duration);

    log!(Level::Debug, "{}", in_memory_file);

    let pca_settings = PcaPlotSettings::default();
    log!(Level::Debug, "Plotting the data!");
    plot::plot_pca(&in_memory_file, None, "test.png", &pca_settings).unwrap();
}

#[test]
fn load_file_with_test_plot_small_lapack() {
    env_logger::init();
    let mut in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad",
        //"/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad"
    )
    .unwrap();
    let n_cells_before = in_memory_file.n_obs();
    log!(Level::Debug, "Performing filtering of all cells");
    single_rust::memory::processing::filter_cells_inplace(
        &mut in_memory_file,
        FlexValue::Absolute(200),
        FlexValue::None,
    )
    .unwrap();
    let n_cells_after = in_memory_file.n_obs();

    log!(
        Level::Debug,
        "Num obs before: {}, Num obs after: {}",
        n_cells_before,
        n_cells_after
    );

    let n_genes_before = in_memory_file.n_vars();
    log!(Level::Debug, "Performing filtering of all genes");
    single_rust::memory::processing::filter_genes_inplace(
        &mut in_memory_file,
        FlexValue::Absolute(3),
        FlexValue::None,
    )
    .unwrap();
    let n_genes_after = in_memory_file.n_vars();

    log!(
        Level::Debug,
        "Num vars before: {}, Num vars after: {}",
        n_genes_before,
        n_genes_after
    );
    let pca_time = Instant::now();
    let f_sel = FeatureSelection::HighlyVariable(25);
    log!(Level::Debug, "{}", in_memory_file);
    single_rust::memory::processing::dim_red::pca_inplace(
        &mut in_memory_file,
        Some(5),
        None,
        Some(true),
        Some(32),
        &f_sel,
        LapackSVD
    )
    .unwrap();
    let duration = pca_time.elapsed();
    log!(Level::Error,"Took time to calculate PCA: {:?}", duration);

    log!(Level::Debug, "{}", in_memory_file);

    let pca_settings = PcaPlotSettings::default();
    log!(Level::Debug, "Plotting the data!");
    plot::plot_pca(&in_memory_file, None, "test.png", &pca_settings).unwrap();
}

#[test]
fn try_subsetting_data_cells() {
    env_logger::init();
    let mut in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad",
        //"/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad"
    )
    .unwrap();
    let n_cells_before = in_memory_file.n_obs();
    log!(Level::Debug, "Performing filtering of all cells");
    single_rust::memory::processing::filter_cells_inplace(
        &mut in_memory_file,
        FlexValue::Absolute(200),
        FlexValue::None,
    )
    .unwrap();
    let n_cells_after = in_memory_file.n_obs();

    log!(
        Level::Debug,
        "Num obs before: {}, Num obs after: {}",
        n_cells_before,
        n_cells_after
    );
}

#[test]
fn try_subsetting_data_genes() {
    env_logger::init();
    let mut in_memory_file = read_h5ad_memory(
        "/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad",
        //"/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad"
    )
    .unwrap();
    log!(Level::Debug, "Var_df: {}", in_memory_file.var().get_data());
    let n_genes_before = in_memory_file.n_vars();
    log!(Level::Debug, "Performing filtering of all genes");
    single_rust::memory::processing::filter_genes_inplace(
        &mut in_memory_file,
        FlexValue::Absolute(15),
        FlexValue::None,
    )
    .unwrap();
    let n_genes_after = in_memory_file.n_vars();

    log!(
        Level::Debug,
        "Num vars before: {}, Num vars after: {}",
        n_genes_before,
        n_genes_after
    );
}
