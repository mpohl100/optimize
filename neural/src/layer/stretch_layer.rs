use crate::layer::dense_layer::DenseLayer;
use crate::layer::dense_layer::MatrixParams;
use crate::utilities::util::WrappedUtils;

use matrix::directory::Directory;
use num_traits::NumCast;

#[derive(Debug, Clone)]
pub struct StretchLayer {
    input_size: usize,
    output_size: usize,
    dense_layers: Vec<DenseLayer>,
    layer_path: Directory,
}

impl StretchLayer {
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
        matrix_params: MatrixParams,
        utils: &WrappedUtils,
    ) -> Self {
        let mut dense_layers = Vec::new();

        let input_size_f64: f64 = NumCast::from(input_size).unwrap_or(0.0);
        let output_size_f64: f64 = NumCast::from(output_size).unwrap_or(0.0);
        let num;
        let (dense_input_size, dense_output_size) = if input_size >= output_size {
            num = NumCast::from((input_size_f64 / output_size_f64).ceil()).unwrap_or(0);
            (num, 1)
        } else {
            num = NumCast::from((output_size_f64 / input_size_f64).ceil()).unwrap_or(0);
            (1, num)
        };

        for i in 0..num {
            let sub_directory = format!("dense_layer_{i}");
            let dense_layer_path = model_directory.expand(&sub_directory);
            let dense_layer = DenseLayer::new(
                dense_input_size,
                dense_output_size,
                dense_layer_path,
                position_in_nn,
                matrix_params,
                utils,
            );
            dense_layers.push(dense_layer);
        }
        Self { input_size, output_size, dense_layers, layer_path: model_directory }
    }
}
