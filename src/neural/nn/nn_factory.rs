use super::{
    directory::Directory,
    neuralnet::{ClassicNeuralNetwork, TrainableClassicNeuralNetwork},
    nn_trait::{WrappedNeuralNetwork, WrappedTrainableNeuralNetwork},
    retry_nn::{RetryNeuralNetwork, TrainableRetryNeuralNetwork},
    shape::NeuralNetworkShape,
};

pub struct NeuralNetworkCreationArguments {
    shape: NeuralNetworkShape,
    levels: Option<i32>,
    model_directory: String,
}

impl NeuralNetworkCreationArguments {
    pub fn new(
        shape: NeuralNetworkShape,
        levels: Option<i32>,
        model_directory: String,
    ) -> Self {
        Self {
            shape,
            levels,
            model_directory,
        }
    }
}

pub fn new_neural_network(
    neural_network_creation_arguments: NeuralNetworkCreationArguments,
) -> WrappedNeuralNetwork {
    match neural_network_creation_arguments.levels {
        Some(levels) => WrappedNeuralNetwork::new(Box::new(RetryNeuralNetwork::new(
            neural_network_creation_arguments.shape,
            levels,
            neural_network_creation_arguments.model_directory,
        ))),
        None => WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(
            neural_network_creation_arguments.shape,
            neural_network_creation_arguments.model_directory,
        ))),
    }
}

pub fn new_trainable_neural_network(
    neural_network_creation_arguments: NeuralNetworkCreationArguments,
) -> WrappedTrainableNeuralNetwork {
    match neural_network_creation_arguments.levels {
        Some(levels) => {
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableRetryNeuralNetwork::new(
                neural_network_creation_arguments.shape,
                levels,
                neural_network_creation_arguments.model_directory,
            )))
        }
        None => WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
            neural_network_creation_arguments.shape,
            Directory::Internal(neural_network_creation_arguments.model_directory),
        ))),
    }
}

pub fn neural_network_from_disk(
    model_directory: String,
) -> WrappedNeuralNetwork {
    // check if model directory contains a directory named primary
    if std::path::Path::new(&format!("{}/primary", model_directory)).exists() {
        return RetryNeuralNetwork::from_disk(
            model_directory,
        );
    }
    WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::from_disk(
        model_directory,
    ).unwrap()))
}

pub fn trainable_neural_network_from_disk(
    model_directory: String,
) -> WrappedTrainableNeuralNetwork {
    // check if model directory contains a directory named primary
    if std::path::Path::new(&format!("{}/primary", model_directory)).exists() {
        return TrainableRetryNeuralNetwork::from_disk(
            model_directory,
        );
    }
    WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::from_disk(
        model_directory,
    ).unwrap()))
}

