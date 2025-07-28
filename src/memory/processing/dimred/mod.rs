pub mod pca;

#[derive(Clone, Debug)]
pub enum FeatureSelectionMethod {
    FullFeatures,
    HighlyVariableSelection(Vec<bool>),
    RandomSelection(usize),
}
