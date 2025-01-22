pub struct StatisticsContainer<I, T>
where
    I: num_traits::PrimInt + num_traits::Unsigned + num_traits::Zero + std::ops::AddAssign,
    T: num_traits::Float + num_traits::NumCast + std::ops::AddAssign + std::iter::Sum + From<I>,
{
    pub num_per_gene: Vec<I>,
    pub expr_per_gene: Vec<T>,
    pub num_per_cell: Vec<I>,
    pub expr_per_cell: Vec<T>,
    pub variance_per_gene: Vec<T>,
    pub variance_per_cell: Vec<T>,
    pub std_dev_per_gene: Vec<T>,
    pub std_dev_per_cell: Vec<T>,
}
