//! # Neural Children Provider
//!
//! This module provides a children provider for neural network-based regret minimization.

use regret::provider::ChildrenProvider;
use regret::regret_node::WrappedRegret;
use regret::user_data::WrappedDecision;

/// Decision data for neural network solver.
#[derive(Debug, Default, Clone)]
pub struct NeuralDecision {
    /// Probability of this decision.
    pub probability: f64,
}

impl regret::user_data::DecisionTrait for NeuralDecision {
    /// Returns the probability of this decision.
    fn get_probability(&self) -> f64 {
        self.probability
    }

    /// Sets the probability of this decision.
    fn set_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability = probability;
    }

    /// Returns a string representation of the decision.
    fn get_data_as_string(&self) -> String {
        format!("Neural Decision with probability: {}", self.probability)
    }
}

/// Children provider for neural network-based decisions.
#[derive(Debug, Clone)]
pub struct NeuralChildrenProvider {
    /// Neural network shape to use for decisions.
    _shape: neural::nn::shape::NeuralNetworkShape,
}

impl NeuralChildrenProvider {
    /// Creates a new neural children provider with the given neural network shape.
    #[must_use]
    pub const fn new(shape: neural::nn::shape::NeuralNetworkShape) -> Self {
        Self { _shape: shape }
    }
}

impl ChildrenProvider<NeuralDecision> for NeuralChildrenProvider {
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        _parents_data: Vec<WrappedDecision<NeuralDecision>>,
    ) -> Vec<WrappedRegret<NeuralDecision>> {
        // TODO: implement
        Vec::new()
    }
}
