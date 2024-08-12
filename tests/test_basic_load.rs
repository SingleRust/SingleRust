use single_rust::io::read_h5ad_memory;



#[test]
fn load_file_test() {
    let in_memory_file = read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/1m_cellxgene_hsa_changed.h5ad").unwrap();
    print!("{}", in_memory_file);
}

#[test]
fn load_file_test_small() {
    let in_memory_file = read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad").unwrap();
    print!("{}", in_memory_file);
}

#[test]
fn load_file_test_print_obs() {
    let in_memory_file = read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad").unwrap();
    let obs = in_memory_file.obs().get_data();
    println!("{}", obs)
}

#[test]
fn load_file_test_print_var() {
    let in_memory_file = read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad").unwrap();
    let var = in_memory_file.var().get_data();
    println!("{}", var)
}

#[test]
fn load_file_test_print_var_names() {
    let in_memory_file = read_h5ad_memory("/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad").unwrap();
    let var = in_memory_file.var_names();
    println!("{:?}", var)
}

