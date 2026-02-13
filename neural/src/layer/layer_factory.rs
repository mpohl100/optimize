use crate::layer::dense_layer::{DenseLayer, TrainableDenseLayer};
use crate::layer::stretch_layer::{StretchLayer, TrainableStretchLayer};
use crate::nn::shape::LayerType;
use crate::utilities::util::WrappedUtils;
use matrix::ai_types::{BiasEntry, NumberEntry, WeightEntry};
use matrix::directory::Directory;

/// Creates a new layer based on the layer type.
///
/// This function dispatches to the appropriate layer constructor based on the
/// `LayerType` variant provided.
///
/// # Arguments
///
/// * `layer_type` - The type of layer to create (Dense or Stretch)
/// * `model_directory` - The directory where the layer's data will be stored
/// * `position_in_nn` - The position of this layer in the neural network
/// * `utils` - Wrapped utilities for resource management
///
/// # Returns
///
/// A boxed layer implementing the `Layer` trait with `NumberEntry` types.
#[must_use]
pub fn new_layer(
    layer_type: &LayerType,
    model_directory: &Directory,
    position_in_nn: usize,
    utils: &WrappedUtils,
) -> Box<dyn crate::layer::layer_trait::Layer<NumberEntry, NumberEntry> + Send> {
    match layer_type {
        LayerType::Dense { input_size, output_size, matrix_params } => Box::new(DenseLayer::new(
            *input_size,
            *output_size,
            model_directory,
            position_in_nn,
            *matrix_params,
            utils,
        )),
        LayerType::Stretch { input_size, output_size, matrix_params } => {
            Box::new(StretchLayer::new(
                *input_size,
                *output_size,
                model_directory.clone(),
                position_in_nn,
                *matrix_params,
                utils,
            ))
        },
    }
}

/// Creates a new trainable layer based on the layer type.
///
/// This function dispatches to the appropriate trainable layer constructor based on the
/// `LayerType` variant provided.
///
/// # Arguments
///
/// * `layer_type` - The type of layer to create (Dense or Stretch)
/// * `model_directory` - The directory where the layer's data will be stored
/// * `position_in_nn` - The position of this layer in the neural network
/// * `utils` - Wrapped utilities for resource management
///
/// # Returns
///
/// A boxed trainable layer implementing the `TrainableLayer` trait.
#[must_use]
pub fn new_trainable_layer(
    layer_type: &LayerType,
    model_directory: &Directory,
    position_in_nn: usize,
    utils: &WrappedUtils,
) -> Box<dyn crate::layer::layer_trait::TrainableLayer<WeightEntry, BiasEntry> + Send> {
    match layer_type {
        LayerType::Dense { input_size, output_size, matrix_params } => {
            Box::new(TrainableDenseLayer::new(
                *input_size,
                *output_size,
                model_directory,
                position_in_nn,
                *matrix_params,
                utils,
            ))
        },
        LayerType::Stretch { input_size, output_size, matrix_params } => {
            Box::new(TrainableStretchLayer::new(
                *input_size,
                *output_size,
                model_directory.clone(),
                position_in_nn,
                *matrix_params,
                utils,
            ))
        },
    }
}
