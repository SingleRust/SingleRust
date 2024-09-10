// --------------------------------- IMPORTANT ---------------------------------
// This version is currently work in progress and we're working on a custom
// library that implements the necessary algorithms that we need. This will
// take some time, until then we rely on nalgebra and smartcore.
// --------------------------------- IMPORTANT ---------------------------------

use anyhow::anyhow;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

use log::{log, Level};
use nshare::{ToNalgebra, ToNdarray2};
use rayon::ThreadPool;
use smartcore::linalg::traits::svd::SVDDecomposable;
use std::cmp::min;

use crate::shared::processing::calculate_svd;

pub struct PCABuilderv2 {
    n_components: Option<usize>,
    center: bool,
    scale: bool,
}

impl Default for PCABuilderv2 {
    fn default() -> Self {
        PCABuilderv2 {
            n_components: None,
            center: true,
            scale: false,
        }
    }
}

impl PCABuilderv2 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }

    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> Pcav2 {
        Pcav2 {
            n_components: self.n_components,
            center: self.center,
            scale: self.scale,
            components: None,
            mean: None,
            std_dev: None,
            explained_variance_ratio: None,
            total_variance: None,
            eigenvalues: None,
        }
    }
}

pub struct Pcav2 {
    n_components: Option<usize>,
    center: bool,
    scale: bool,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    std_dev: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    total_variance: Option<f64>,
    eigenvalues: Option<Array1<f64>>,
}

impl Pcav2 {
    pub fn fit(&mut self, data: &Array2<f64>, thread_pool: &ThreadPool) -> anyhow::Result<()> {
        log!(Level::Debug, "Starting the fitting process");
        let (nrows, ncols) = data.dim();
        log!(Level::Debug, "Aquired dimensions");
        let n_components = self.n_components.unwrap_or(min(nrows, ncols));
        log!(Level::Debug, "Unwraped and aquired components...");

        // Center and scale the data
        log!(Level::Debug, "Centering the data....");
        let mut centered_data = data.to_owned();
        log!(Level::Debug, "Centered the data successfully");
        if self.center || self.scale {
            log!(
                Level::Debug,
                "Calculating the mean if the axis in a thread pool"
            );
            let mean = thread_pool.install(|| data.mean_axis(Axis(0)).unwrap());
            log!(Level::Debug, "Calculated successfully");
            log!(Level::Debug, "Calculating standard deviation");
            let std_dev = if self.scale {
                thread_pool.install(|| data.std_axis(Axis(0), 0.))
            } else {
                Array1::ones(ncols)
            };
            log!(Level::Debug, "Calcualted successfully");

            thread_pool.install(|| {
                if self.center {
                    log!(Level::Debug, "Performing centering");
                    centered_data.axis_iter_mut(Axis(0)).for_each(|mut row| {
                        row.zip_mut_with(&mean, |x, &m| *x -= m);
                    });
                    log!(Level::Debug, "Centered successfully");
                }
                if self.scale {
                    log!(Level::Debug, "Performing scaling");
                    centered_data.axis_iter_mut(Axis(0)).for_each(|mut row| {
                        row.zip_mut_with(&std_dev, |x, &s| *x /= s);
                    });
                    log!(Level::Debug, "Scaled successfully");
                }
            });

            self.mean = Some(mean);
            self.std_dev = Some(std_dev);
        } else {
            self.mean = Some(Array1::zeros(ncols));
            self.std_dev = Some(Array1::ones(ncols));
        }

        log!(Level::Debug, "Calculaing SVD now");
        // Perform SVD
        //let svd = thread_pool.install(|| centered_data.svd(true, false))?;
        let (s, vt) = super::calculate_svd(centered_data.view(), thread_pool)?;
        log!(Level::Debug, "Calculated svd in thread pool successfully");

        // Calculate explained variance and ratio
        log!(Level::Debug, "Calculating eigenvalues and ratio");
        let (eigenvalues, total_variance, explained_variance_ratio) = thread_pool.install(|| {
            let eigenvalues = &s.mapv(|x| x * x / (nrows - 1) as f64);
            let total_variance = eigenvalues.sum();
            let explained_variance_ratio = eigenvalues.mapv(|x| x / total_variance);
            (
                eigenvalues.to_owned(),
                total_variance,
                explained_variance_ratio,
            )
        });
        log!(Level::Debug, "Successfully computed....");

        // Store results
        log!(Level::Debug, "Storing the results....");
        self.components = Some(vt.slice(s![.., ..n_components]).to_owned());
        self.explained_variance_ratio = Some(
            explained_variance_ratio
                .slice(s![..n_components])
                .to_owned(),
        );
        self.total_variance = Some(total_variance);
        self.eigenvalues = Some(eigenvalues);

        Ok(())
    }

    pub fn transform(
        &self,
        data: &Array2<f64>,
        thread_pool: &ThreadPool,
    ) -> anyhow::Result<Array2<f64>> {
        if self.components.is_none() {
            return Err(anyhow!("PCA model has not been fitted. Call fit() first."));
        }

        let components = self.components.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();
        let std_dev = self.std_dev.as_ref().unwrap();

        let mut centered_data = data.to_owned();
        thread_pool.install(|| {
            if self.center {
                centered_data.axis_iter_mut(Axis(0)).for_each(|mut row| {
                    row.zip_mut_with(mean, |x, &m| *x -= m);
                })
            }

            if self.scale {
                centered_data.axis_iter_mut(Axis(0)).for_each(|mut row| {
                    row.zip_mut_with(std_dev, |x, &s| *x /= s);
                })
            }
        });

        Ok(thread_pool.install(|| centered_data.dot(components)))
    }

    pub fn fit_transform(
        &mut self,
        data: &Array2<f64>,
        thread_pool: &ThreadPool,
    ) -> anyhow::Result<Array2<f64>> {
        self.fit(data, thread_pool)?;
        self.transform(data, thread_pool)
    }

    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    pub fn compute_loadings(&self) -> Option<Array2<f64>> {
        if let (Some(components), Some(std_dev)) = (&self.components, &self.std_dev) {
            Some(
                components.t().to_owned()
                    * std_dev
                        .broadcast((components.ncols(), std_dev.len()))
                        .unwrap(),
            )
        } else {
            None
        }
    }
}
