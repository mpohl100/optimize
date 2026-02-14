//! # Neural State
//!
//! This module defines the state trait and neural state implementation.

use neural::nn::shape::NeuralNetworkShape;

/// Trait for state types used in neural network solving.
pub trait StateTrait: Default + Clone + std::fmt::Debug {
    /// Returns a string representation of the state.
    fn get_data_as_string(&self) -> String;
}

/// Neural network state containing the shape.
#[derive(Debug, Clone, Default)]
pub struct NeuralState {
    /// The neural network shape.
    shape: NeuralNetworkShape,
}

impl NeuralState {
    /// Creates a new neural state with the given shape.
    #[must_use]
    pub const fn new(shape: NeuralNetworkShape) -> Self {
        Self { shape }
    }

    /// Returns the neural network shape.
    #[must_use]
    pub const fn get_shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }
}

impl StateTrait for NeuralState {
    fn get_data_as_string(&self) -> String {
        format!("NeuralState with {} layers", self.shape.num_layers())
    }
}
