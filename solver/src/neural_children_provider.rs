//! # Neural Children Provider
//!
//! This module provides a children provider for neural network-based regret minimization.

use neural::nn::shape::NeuralNetworkShape;
use regret::provider::ChildrenProvider;
use regret::regret_node::WrappedRegret;
use regret::user_data::{DecisionTrait, WrappedDecision};

use crate::neural_state::{NeuralState, StateTrait};

/// User data for neural network solver.
#[derive(Debug, Clone)]
pub struct NeuralUserData {
    /// The neural state.
    state: NeuralState,
    /// Probability of this decision.
    probability: f64,
}

impl Default for NeuralUserData {
    fn default() -> Self {
        Self { state: NeuralState::default(), probability: 0.0 }
    }
}

impl NeuralUserData {
    /// Creates a new neural user data with the given state.
    #[must_use]
    pub const fn new(state: NeuralState) -> Self {
        Self { state, probability: 0.0 }
    }

    /// Returns the neural state.
    #[must_use]
    pub fn get_state(&self) -> NeuralState {
        self.state.clone()
    }
}

impl DecisionTrait for NeuralUserData {
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
        format!("State: {}, Probability: {}", self.state.get_data_as_string(), self.probability)
    }
}

/// Children provider for neural network-based decisions.
#[derive(Debug, Clone)]
pub struct NeuralChildrenProvider {
    /// Neural network shape to use for decisions.
    #[allow(dead_code)]
    shape: NeuralNetworkShape,
}

impl NeuralChildrenProvider {
    /// Creates a new neural children provider with the given neural network shape.
    #[must_use]
    pub const fn new(shape: NeuralNetworkShape) -> Self {
        Self { shape }
    }
}

impl ChildrenProvider<NeuralUserData> for NeuralChildrenProvider {
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        _parents_data: Vec<WrappedDecision<NeuralUserData>>,
    ) -> Vec<WrappedRegret<NeuralUserData>> {
        // TODO: implement
        Vec::new()
    }
}
