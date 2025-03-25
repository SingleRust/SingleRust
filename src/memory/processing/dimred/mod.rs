pub mod pca;

pub enum FeatureSelectionMethod {
    FullFeatures,
    HighlyVariableSelection(Vec<bool>),
    RandomSelection(usize)
}