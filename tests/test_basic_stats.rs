use std::time::Instant;

use anndata_memory::IMAnnData;
use single_rust::{io::read_h5ad_memory, memory::statistics::compute_number, Direction};


fn load_small_memory() -> anyhow::Result<IMAnnData> {
    read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad")
}

fn load_big_memory() -> anyhow::Result<IMAnnData> {
    read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad")
}


#[test]
fn test_calc_row_number_small() {
    let data = load_small_memory().unwrap();

    let start = Instant::now();
    let res = compute_number(data, Direction::Row).unwrap();
    
    let duration = start.elapsed();
    
    println!("Result length: {}", res.len());
    println!("Time taken: {:?}", duration);
}

#[test]
fn test_calc_obs_number_small() {
    let data = load_small_memory().unwrap();

    let start = Instant::now();
    let res = compute_number(data, Direction::Column).unwrap();
    
    let duration = start.elapsed();
    
    println!("Result length: {}", res.len());
    println!("Time taken: {:?}", duration);
}


#[test]
fn test_calc_row_number_big() {
    let data = load_big_memory().unwrap();
    println!("loaded file!");
    let start = Instant::now();
    let res = compute_number(data, Direction::Row).unwrap();
    
    let duration = start.elapsed();
    
    println!("Result length: {}", res.len());
    println!("Time taken: {:?}", duration);
}

#[test]
fn test_calc_obs_number_big() {
    let data = load_big_memory().unwrap();

    let start = Instant::now();
    let res = compute_number(data, Direction::Column).unwrap();
    
    let duration = start.elapsed();
    
    println!("Result length: {}", res.len());
    println!("Time taken: {:?}", duration);
}