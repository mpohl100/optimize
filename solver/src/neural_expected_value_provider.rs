//! # Neural Expected Value Provider
//!
//! This module provides an expected value provider for neural network-based regret minimization.

use regret::provider::ExpectedValueProvider;
use regret::user_data::WrappedDecision;

use crate::neural_children_provider::NeuralUserData;

/// Expected value provider for neural network-based decisions.
#[derive(Debug, Clone)]
pub struct NeuralExpectedValueProvider {}

impl NeuralExpectedValueProvider {
    /// Creates a new neural expected value provider.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

impl Default for NeuralExpectedValueProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpectedValueProvider<NeuralUserData> for NeuralExpectedValueProvider {
    /// Returns the expected value for the given parent data.
    fn get_expected_value(
        &self,
        _parents_data: Vec<WrappedDecision<NeuralUserData>>,
    ) -> f64 {
        // TODO: implement
        0.0
    }
}
