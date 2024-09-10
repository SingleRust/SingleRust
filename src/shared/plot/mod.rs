mod settings;

use std::{collections::HashSet, path::Path};

use anndata::{data::DynArray, ArrayData};
use ndarray::{s, ArrayD, ArrayView2, Ix2};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, Circle, IntoDrawingArea},
    style::{Color, IntoFont, Palette, Palette99, RGBAColor, BLACK},
};
use log::{log, Level};
pub use settings::PcaPlotSettings;

pub fn plot_pca_array_data<P: AsRef<Path>>(
    pca_coords: &ArrayData,
    colors: Option<&[String]>,
    output_path: P,
    settings: &PcaPlotSettings,
) -> anyhow::Result<()> {
    log!(Level::Debug, "Trying to extract array from arrayData");
    match pca_coords {
        ArrayData::Array(array) => plot_pca_sub_array(array, colors, output_path, settings),
        ArrayData::CsrMatrix(_) => todo!(),
        ArrayData::CsrNonCanonical(_) => todo!(),
        ArrayData::CscMatrix(_) => todo!(),
        ArrayData::DataFrame(_) => todo!(),
    }
}

fn plot_pca_sub_array<P: AsRef<Path>>(
    dyn_array: &DynArray,
    colors: Option<&[String]>,
    output_path: P,
    settings: &PcaPlotSettings,
) -> anyhow::Result<()> {
    // check for dimensionality first
    log!(Level::Debug, "Extracting F32 and F64 datatypes");
    match dyn_array {
        DynArray::F32(arr) => plot_pca_sub_multi_array(arr, colors, output_path, settings),
        DynArray::F64(arr) => plot_pca_sub_multi_array(arr, colors, output_path, settings),
        _ => Err(anyhow::anyhow!(
            "This type is not supported when plotting PCA using a coordinate system"
        )),
    }
}

fn plot_pca_sub_multi_array<T: num_traits::Float + std::fmt::Debug, P: AsRef<Path>>(
    dyn_array: &ArrayD<T>,
    colors: Option<&[String]>,
    output_path: P,
    settings: &PcaPlotSettings,
) -> anyhow::Result<()> {
    log!(Level::Debug, "Checking for correct dimensionality");
    let shape = dyn_array.shape();

    match shape.len() {
        1 => Err(anyhow::anyhow!(
            "The ArrayD must have at least two dimensions!"
        )),
        2 => {
            log!(Level::Debug, "Setting correct dimensionaloty");
            let d = dyn_array.view().into_dimensionality::<Ix2>()?;
            log!(Level::Debug, "Number of rows: {}", d.nrows());
            log!(Level::Debug, "Starting to plot data");
            plot_pca_numeric(d, colors, output_path, settings)
        }
        _ => Err(anyhow::anyhow!("Unsupported amount of array dimensions ")),
    }
}

pub(crate) fn plot_pca_numeric<T: num_traits::Float + std::fmt::Debug, P: AsRef<Path>>(
    pca_coords: ArrayView2<T>,
    colors: Option<&[String]>,
    output_path: P,
    settings: &PcaPlotSettings,
) -> anyhow::Result<()> {

    log!(Level::Debug, "PCA coordinates shape: {:?}", pca_coords.shape());
    
    if pca_coords.is_empty() {
        return Err(anyhow::anyhow!("PCA coordinates array is empty"));
    }

    let x_column = pca_coords.column(settings.x_axis);
    let y_column = pca_coords.column(settings.y_axis);

    log!(Level::Debug, "First few x values: {:?}", &x_column.slice(s![..5]));
    log!(Level::Debug, "First few y values: {:?}", &y_column.slice(s![..5]));

    let x_min = x_column.fold(T::infinity(), |a, &b| if b.is_finite() { a.min(b) } else { a });
    let x_max = x_column.fold(T::neg_infinity(), |a, &b| if b.is_finite() { a.max(b) } else { a });
    let y_min = y_column.fold(T::infinity(), |a, &b| if b.is_finite() { a.min(b) } else { a });
    let y_max = y_column.fold(T::neg_infinity(), |a, &b| if b.is_finite() { a.max(b) } else { a });

    log!(Level::Debug, "X_min: {:?}, X_max: {:?}, y_min: {:?}, y_max: {:?}", 
        x_min.to_f64(), x_max.to_f64(), y_min.to_f64(), y_max.to_f64());

    if !x_min.is_finite() || !x_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
        return Err(anyhow::anyhow!("Unable to determine finite min/max values for plotting"));
    }

    let x_min = x_min.to_f64().unwrap();
    let x_max = x_max.to_f64().unwrap();
    let y_min = y_min.to_f64().unwrap();
    let y_max = y_max.to_f64().unwrap();
    
    // Base drawing board
    log!(Level::Debug, "Creating the drawing area");
    let root = BitMapBackend::new(output_path.as_ref(), (settings.width, settings.height))
        .into_drawing_area();
    log!(Level::Debug, "Filling the root of the plot");
    // fill with background color
    root.fill(&settings.background_color)?;

    log!(Level::Debug, "Creating the chart context");
    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    log!(Level::Debug, "Configuring mesh of the chart");
    chart
        .configure_mesh()
        .x_desc(&settings.x_label)
        .y_desc(&settings.y_label)
        .draw()?;

    log!(Level::Debug, "Extracting the unique colors from the dataset!");
    // Prepare unique colors if colors are provided
    let unique_colors = colors.as_ref().map(|c| {
        c.iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
    });

    
    // Plot the points
    if let Some(color_vec) = &colors {
        log!(Level::Debug, "Drawing colored circles");
        for (i, color_name) in color_vec.iter().enumerate() {
            let x = pca_coords[(i, settings.x_axis)].to_f64().unwrap();
            let y = pca_coords[(i, settings.y_axis)].to_f64().unwrap();

            let color_index = unique_colors
                .as_ref()
                .unwrap()
                .iter()
                .position(|c| c == color_name)
                .unwrap();
            let color = Palette99::pick(color_index).to_rgba();

            chart.draw_series(std::iter::once(Circle::new(
                (x, y),
                settings.point_size,
                color.filled(),
            )))?;
        }
    } else {
        log!(Level::Debug, "Drawing circles without color");
        for i in 0..pca_coords.nrows() {
            let x = pca_coords[(i, settings.x_axis)].to_f64().unwrap();
            let y = pca_coords[(i, settings.y_axis)].to_f64().unwrap();
            chart.draw_series(std::iter::once(Circle::new(
                (x, y),
                settings.point_size,
                settings.point_color.filled(),
            )))?;
        }
    }

    // Add a color legend if colors are provided
    if let Some(unique) = &unique_colors {
        chart
            .configure_series_labels()
            .background_style(settings.background_color.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        for (i, color_value) in unique.iter().enumerate() {
            let color = Palette99::pick(i).to_rgba();
            chart
                .draw_series(std::iter::once(Circle::new(
                    (x_min, y_max - (i as f64) * 0.1 * (y_max - y_min)),
                    5,
                    color.filled(),
                )))?
                .label(color_value)
                .legend(move |(x, y)| Circle::new((x, y), 5, color.filled()));
        }
    }

    root.present()?;

    Ok(())
}
