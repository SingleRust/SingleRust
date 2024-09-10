use plotters::style::RGBAColor;

pub struct PcaPlotSettings {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub background_color: RGBAColor,
    pub point_color: RGBAColor,
    pub point_size: u32,
    pub x_axis: usize,
    pub y_axis: usize,
}

impl Default for PcaPlotSettings {
    fn default() -> Self {
        PcaPlotSettings {
            width: 800,
            height: 600,
            title: "PCA Plot".to_string(),
            x_label: "PC1".to_string(),
            y_label: "PC2".to_string(),
            background_color: RGBAColor(255, 255, 255, 1.0),  // White
            point_color: RGBAColor(0, 0, 255, 0.5),  // Semi-transparent blue
            point_size: 2,
            x_axis: 0,
            y_axis: 1,
        }
    }
}

impl PcaPlotSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    pub fn with_labels(mut self, x_label: &str, y_label: &str) -> Self {
        self.x_label = x_label.to_string();
        self.y_label = y_label.to_string();
        self
    }

    pub fn with_background_color(mut self, color: RGBAColor) -> Self {
        self.background_color = color;
        self
    }

    pub fn with_point_color(mut self, color: RGBAColor) -> Self {
        self.point_color = color;
        self
    }

    pub fn with_point_size(mut self, size: u32) -> Self {
        self.point_size = size;
        self
    }

    pub fn with_axes(mut self, x_axis: usize, y_axis: usize) -> Self {
        self.x_axis = x_axis;
        self.y_axis = y_axis;
        self
    }
}