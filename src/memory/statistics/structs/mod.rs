pub struct StatisticsContainer {
    pub num_per_gene: Vec<u32>,
    pub expr_per_gene: Vec<f64>,
    pub num_per_cell: Vec<u32>,
    pub expr_per_cell: Vec<f64>,
    pub variance_per_gene: Vec<f64>,
    pub variance_per_cell: Vec<f64>,
    pub std_dev_per_gene: Vec<f64>,
    pub std_dev_per_cell: Vec<f64>
}

