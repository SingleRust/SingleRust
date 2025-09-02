use anndata::data::SelectInfoElem;
use log::{log, Level};
use ndarray::{ArrayView, Ix1};
use polars::prelude::DataType;

pub fn get_select_info_obs(
    obs_mask: Option<ArrayView<'_, bool, Ix1>>,
) -> anyhow::Result<Vec<SelectInfoElem>> {
    // Convert masks to indices
    let obs_indices = obs_mask.map(|mask| {
        mask.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect::<Vec<_>>()
    });

    // Create a selection based on the masks
    let mut selection = vec![SelectInfoElem::full(), SelectInfoElem::full()];
    if let Some(obs_idx) = obs_indices {
        selection[0] = SelectInfoElem::Index(obs_idx);
    }

    Ok(selection)
}

pub fn get_select_info_vars(
    vars_mask: Option<ArrayView<'_, bool, Ix1>>,
) -> anyhow::Result<Vec<SelectInfoElem>> {
    // Convert mask to indices
    let vars_indices = vars_mask.map(|mask| {
        mask.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect::<Vec<_>>()
    });

    // Create a selection based on the mask
    let mut selection = vec![SelectInfoElem::full(), SelectInfoElem::full()];
    if let Some(vars_idx) = vars_indices {
        log!(Level::Debug, "VarIDx length: {}", vars_idx.len());
        selection[1] = SelectInfoElem::Index(vars_idx);
    }

    Ok(selection)
}

#[derive(Debug)]
pub enum FlavorType {
    Seurat,
    CellRanger,
    SVR,
}

pub struct HVGParams {
    pub min_mean: f64,
    pub max_mean: f64,
    pub min_dispersion: f64,
    pub max_dispersion: f64,
    pub n_bins: usize,
    pub n_top_genes: Option<usize>,
    pub flavor: FlavorType,
    pub span: f64,
    pub batch_key: Option<String>,
}

impl Default for HVGParams {
    fn default() -> Self {
        HVGParams {
            min_mean: 0.0125,
            max_mean: 3.0,
            min_dispersion: 0.5,
            max_dispersion: f64::INFINITY,
            n_bins: 20,
            n_top_genes: None,
            flavor: FlavorType::Seurat,
            span: 0.3,
            batch_key: None,
        }
    }
}

pub fn standardize_log(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma == 0.0 {
        return 0.0;
    }
    (x - mu) / sigma
}

pub fn standardize_log_form_vec(vec: &[f64]) -> Vec<f64> {
    // Filter out non-finite values for statistics calculation
    let finite_values: Vec<f64> = vec.iter().filter(|x| x.is_finite()).copied().collect();

    let n = finite_values.len() as f64;
    if n <= 1.0 {
        // If we have too few finite values, return zeros
        println!(
            "DEBUG: Too few finite values ({}) for standardization, returning zeros",
            n
        );
        return vec![0.0; vec.len()];
    }

    let mu: f64 = finite_values.iter().sum::<f64>() / n;
    let sigma = (finite_values.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

    // Debug the standardization process
    let finite_count = vec.iter().filter(|x| x.is_finite()).count();
    let nan_count = vec.iter().filter(|x| x.is_nan()).count();
    let inf_count = vec.iter().filter(|x| x.is_infinite()).count();
    println!(
        "DEBUG Standardization: n={}, mu={:.6}, sigma={:.6}, finite={}, nan={}, inf={}",
        vec.len(),
        mu,
        sigma,
        finite_count,
        nan_count,
        inf_count
    );

    if !sigma.is_finite() || sigma == 0.0 {
        println!("DEBUG: Sigma is problematic, returning zeros");
        return vec![0.0; vec.len()];
    }

    // Standardize all values, but only finite ones get proper standardization
    vec.iter()
        .map(|&x| {
            if x.is_finite() {
                standardize_log(x, mu, sigma)
            } else {
                0.0 // Non-finite values get zero standardized score
            }
        })
        .collect()
}

#[allow(dead_code)]
fn normalize_per_bin(
    log_means: &[f64],
    log_dispersions: &[f64],
    n_bins: usize,
) -> anyhow::Result<Vec<f64>> {
    let min_mean = log_means.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_mean = log_means.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let bin_width = (max_mean - min_mean) / n_bins as f64;

    let mut norm_dispersions = vec![0.0; log_means.len()];
    let mut bin_counts = vec![0; n_bins];
    let mut bin_disp_means = vec![0.0; n_bins];
    let mut bin_disp_stds = vec![0.0; n_bins];

    for i in 0..log_means.len() {
        let bin_idx = if log_means[i] == max_mean {
            n_bins - 1
        } else {
            ((log_means[i] - min_mean) / bin_width) as usize
        };

        bin_counts[bin_idx] += 1;
        bin_disp_means[bin_idx] += log_dispersions[i];
    }

    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            bin_disp_means[i] /= bin_counts[i] as f64;
        }
    }

    for i in 0..log_means.len() {
        let bin_idx = if log_means[i] == max_mean {
            n_bins - 1
        } else {
            ((log_means[i] - min_mean) / bin_width) as usize
        };
        if bin_counts[bin_idx] > 1 {
            bin_disp_stds[bin_idx] += (log_dispersions[i] - bin_disp_means[bin_idx]).powi(2);
        }
    }

    for i in 0..n_bins {
        if bin_counts[i] > 1 {
            bin_disp_stds[i] = (bin_disp_stds[i] / (bin_counts[i] - 1) as f64).sqrt();
        }
    }

    for i in 0..log_means.len() {
        let bin_idx = if log_means[i] == max_mean {
            n_bins - 1
        } else {
            ((log_means[i] - min_mean) / bin_width) as usize
        };

        if bin_disp_stds[bin_idx] > 0.0 {
            norm_dispersions[i] =
                (log_dispersions[i] - bin_disp_means[bin_idx]) / bin_disp_stds[bin_idx];
        }
    }

    Ok(norm_dispersions)
}

pub fn _normalize_by_batch(
    log_means: &[f64],
    log_dispersions: &[f64],
    batch_col: &polars::prelude::Column,
    n_bins: usize,
) -> anyhow::Result<Vec<f64>> {
    let mut unique_batches = Vec::new();
    if let DataType::String = batch_col.dtype() {
        unique_batches = batch_col.str()?.iter().flatten().collect::<Vec<&str>>();
    }

    let mut norm_dispersions = vec![0.0; log_means.len()];
    for batch in unique_batches {
        let batch_mask: Vec<bool> = batch_col
            .str()?
            .into_iter()
            .map(|x| x == Some(batch))
            .collect();

        let batch_size = batch_mask.iter().filter(|&&x| x).count();

        if batch_size > 0 {
            let batch_norm = normalize_per_bin(
                &log_means
                    .iter()
                    .zip(batch_mask.iter())
                    .filter(|(_, &mask)| mask)
                    .map(|(&x, _)| x)
                    .collect::<Vec<_>>(),
                &log_dispersions
                    .iter()
                    .zip(batch_mask.iter())
                    .filter(|(_, &mask)| mask)
                    .map(|(&x, _)| x)
                    .collect::<Vec<_>>(),
                n_bins,
            )?;

            let mut j = 0;
            for i in 0..log_means.len() {
                if batch_mask[i] {
                    norm_dispersions[i] = batch_norm[j];
                    j += 1;
                }
            }
        }
    }
    Ok(norm_dispersions)
}

pub fn fit_svr(x: &[f64], y: &[f64]) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let n = x.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }

    let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let x_norm: Vec<f64> = x.iter().map(|&xi| (xi - x_min) / (x_max - x_min)).collect();

    let gamma = 1.0 / n as f64;
    let mut kernel = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let diff = x_norm[i] - x_norm[j];
            kernel[j][i] = (-gamma * diff.powi(2)).exp();
        }
    }

    let lambda = 1.0; // Regularization parameter
    let mut alpha = vec![0.0; n];

    // Solve (K + λI)α = y using simple iterative method
    for _ in 0..100 {
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += kernel[j][i] * alpha[j];
                }
            }
            alpha[i] = (y[i] - sum) / (kernel[i][i] + lambda);
        }
    }

    let mut y_pred = vec![0.0; n];
    for i in 0..n {
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            y_pred[i] += alpha[j] * kernel[i][j];
        }
    }

    let residuals: Vec<f64> = y
        .iter()
        .zip(y_pred.iter())
        .map(|(&yi, &yp)| yi - yp)
        .collect();

    Ok((residuals, y_pred))
}

pub fn _get_mean_bins(
    log_means: &[f64],
    n_bins: usize,
) -> anyhow::Result<(Vec<usize>, Vec<usize>)> {
    // Use quantile-based binning instead of equal-width binning
    // This ensures each bin has roughly the same number of genes

    let mut sorted_means: Vec<(usize, f64)> = log_means
        .iter()
        .enumerate()
        .map(|(i, &mean)| (i, mean))
        .collect();

    // Sort by log_means values
    sorted_means.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_genes = log_means.len();
    let genes_per_bin = n_genes / n_bins;
    let remainder = n_genes % n_bins;

    let mut bin_indices = vec![0; log_means.len()];
    let mut mean_bins = vec![0; n_bins];

    let mut current_gene_idx = 0;

    (0..n_bins).for_each(|bin_idx| {
        // Calculate how many genes should be in this bin
        // First 'remainder' bins get one extra gene
        let genes_in_this_bin = if bin_idx < remainder {
            genes_per_bin + 1
        } else {
            genes_per_bin
        };

        // Assign genes to this bin
        for _ in 0..genes_in_this_bin {
            if current_gene_idx < sorted_means.len() {
                let original_idx = sorted_means[current_gene_idx].0;
                bin_indices[original_idx] = bin_idx;
                current_gene_idx += 1;
            }
        }

        mean_bins[bin_idx] = genes_in_this_bin;
    });

    Ok((mean_bins, bin_indices))
}

pub fn _calculate_dispersion_stats(
    log_dispersions: &[f64],
    bin_indices: &[usize],
    mean_bins: &[usize],
) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let n_bins = mean_bins.len();
    let mut bin_means = vec![0.0; n_bins];
    let mut bin_stds = vec![0.0; n_bins];
    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_sum_squares = vec![0.0; n_bins];

    for (i, &bin_idx) in bin_indices.iter().enumerate() {
        let disp = log_dispersions[i];
        if !disp.is_nan() {
            bin_sums[bin_idx] += disp;
            bin_sum_squares[bin_idx] += disp * disp;
        }
    }

    for bin_idx in 0..n_bins {
        let count = mean_bins[bin_idx] as f64;
        if count > 0.0 {
            bin_means[bin_idx] = bin_sums[bin_idx] / count;

            if count > 1.0 {
                let variance =
                    (bin_sum_squares[bin_idx] - bin_sums[bin_idx].powi(2) / count) / (count - 1.0);

                // Add small epsilon to prevent zero standard deviation
                let min_variance = 1e-12;
                bin_stds[bin_idx] = (variance.max(min_variance)).sqrt();
            } else {
                bin_stds[bin_idx] = f64::NAN;
            }
        } else {
            bin_means[bin_idx] = f64::NAN;
            bin_stds[bin_idx] = f64::NAN;
        }
    }

    Ok((bin_means, bin_stds))
}
