//! # Neural State
//!
//! This module defines the state trait and neural state implementation.

use neural::nn::nn_trait::WrappedTrainableNeuralNetwork;
use neural::nn::shape::NeuralNetworkShape;

/// Trait for state types used in neural network solving.
pub trait StateTrait: Default + Clone + std::fmt::Debug {
    /// Returns a string representation of the state.
    fn get_data_as_string(&self) -> String;
}

/// Neural network state containing the shape and optional neural network.
#[derive(Debug, Clone, Default)]
pub struct NeuralState {
    /// The neural network shape.
    shape: NeuralNetworkShape,
    /// The trainable neural network, if available.
    neural_network: Option<WrappedTrainableNeuralNetwork>,
}

impl NeuralState {
    /// Creates a new neural state with the given shape.
    #[must_use]
    pub const fn new(shape: NeuralNetworkShape) -> Self {
        Self { shape, neural_network: None }
    }

    /// Creates a new neural state with the given shape and neural network.
    #[must_use]
    pub const fn new_with_nn(
        shape: NeuralNetworkShape,
        neural_network: WrappedTrainableNeuralNetwork,
    ) -> Self {
        Self { shape, neural_network: Some(neural_network) }
    }

    /// Returns the neural network shape.
    #[must_use]
    pub const fn get_shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    /// Returns a reference to the neural network, if available.
    #[must_use]
    pub const fn get_neural_network(&self) -> &Option<WrappedTrainableNeuralNetwork> {
        &self.neural_network
    }

    /// Sets the neural network.
    pub fn set_neural_network(
        &mut self,
        neural_network: WrappedTrainableNeuralNetwork,
    ) {
        self.neural_network = Some(neural_network);
    }
}

impl StateTrait for NeuralState {
    fn get_data_as_string(&self) -> String {
        format!("NeuralState with {} layers", self.shape.num_layers())
    }
}
