use super::shape::NeuralNetworkShape;
use super::nn_trait::WrappedNeuralNetwork;

use crate::gen::pheno::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::neural::nn::shape::LayerShape;
use crate::neural::nn::shape::LayerType;
use crate::neural::nn::neuralnet::ClassicNeuralNetwork;

pub struct RetryNeuralNetwork{
    primary_nn: WrappedNeuralNetwork,
    backup_nn: WrappedNeuralNetwork,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    // The actual shape of the neural network that have some internal dimensions to decide whether to invoke the backup_nn
    actual_shape: NeuralNetworkShape
}

impl RetryNeuralNetwork {
    pub fn new(shape: NeuralNetworkShape, internal_model_directory: String) -> Self
    {
        let actual_shape = add_internal_dimensions(shape.clone());
        let primary_nn = WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(actual_shape.clone(), internal_model_directory.clone())));
        let backup_nn = WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(shape.clone(), internal_model_directory.clone())));
        Self {
            primary_nn,
            backup_nn,
            shape,
            actual_shape
        }
    }
}

fn add_internal_dimensions(shape: NeuralNetworkShape) -> NeuralNetworkShape {
    // Add internal dimensions to the shape
    let mut annotated_shape = AnnotatedNeuralNetworkShape::new(shape.clone());
    let first_layer = shape.layers.first().unwrap();

    // Add internal dimensions to the first layer
    let internal_layer = first_layer.clone();
    let new_dense_layer_type = LayerShape{ layer_type: LayerType::Dense {
        input_size: internal_layer.input_size(),
        output_size: internal_layer.output_size() + 1,
    }, activation: internal_layer.activation.clone() };
    annotated_shape.change_layer(0, new_dense_layer_type);

    // Add internal dimensions to the rest of the layers
    for (i, layer) in shape.layers.iter().skip(1).enumerate() {
        // Add internal dimensions to the layer
        let internal_layer = layer.clone();
        let new_dense_layer_type = LayerShape{ layer_type: LayerType::Dense {
            input_size: internal_layer.input_size() + 1,
            output_size: internal_layer.output_size() + 1,
        }, activation: internal_layer.activation.clone() };
        annotated_shape.change_layer(i, new_dense_layer_type);
    }
    annotated_shape.to_neural_network_shape()
}