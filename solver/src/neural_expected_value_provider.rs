//! # Neural Expected Value Provider
//!
//! This module provides an expected value provider for neural network-based regret minimization.

use regret::provider::ExpectedValueProvider;
use regret::user_data::WrappedDecision;

use crate::neural_children_provider::NeuralDecision;

/// Expected value provider for neural network-based decisions.
#[derive(Debug, Clone)]
pub struct NeuralExpectedValueProvider {
    /// Neural network shape to use for expected value calculation.
    _shape: neural::nn::shape::NeuralNetworkShape,
}

impl NeuralExpectedValueProvider {
    /// Creates a new neural expected value provider with the given neural network shape.
    #[must_use]
    pub const fn new(shape: neural::nn::shape::NeuralNetworkShape) -> Self {
        Self { _shape: shape }
    }
}

impl ExpectedValueProvider<NeuralDecision> for NeuralExpectedValueProvider {
    /// Returns the expected value for the given parent data.
    fn get_expected_value(
        &self,
        _parents_data: Vec<WrappedDecision<NeuralDecision>>,
    ) -> f64 {
        // TODO: implement
        0.0
    }
}
