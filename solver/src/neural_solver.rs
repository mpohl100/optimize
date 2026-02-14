//! # Neural Solver
//!
//! This module provides a solver that combines neural networks with regret minimization.

use neural::nn::shape::NeuralNetworkShape;
use regret::provider::{Provider, ProviderType, WrappedChildrenProvider, WrappedProvider};
use regret::regret_node::RegretNode;

use crate::neural_children_provider::NeuralChildrenProvider;
use crate::neural_expected_value_provider::NeuralExpectedValueProvider;

/// A solver that uses neural networks and regret minimization.
#[derive(Debug)]
pub struct NeuralSolver {
    /// The neural network shape to use for solving.
    shape: NeuralNetworkShape,
}

impl NeuralSolver {
    /// Creates a new neural solver with the given neural network shape.
    #[must_use]
    pub const fn new(shape: NeuralNetworkShape) -> Self {
        Self { shape }
    }

    /// Solves the problem using regret minimization with the neural network.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - The number of iterations to run the regret minimization algorithm.
    /// * `do_randomize_children` - Whether to randomize children during solving.
    pub fn solve(
        &self,
        num_iterations: usize,
        do_randomize_children: bool,
    ) {
        // Create the children provider and expected value provider
        let children_provider = NeuralChildrenProvider::new(self.shape.clone());
        let _expected_value_provider = NeuralExpectedValueProvider::new(self.shape.clone());

        // Create the provider type with children provider
        let provider_type =
            ProviderType::Children(WrappedChildrenProvider::new(Box::new(children_provider)));

        // Create the wrapped provider
        let provider = WrappedProvider::new(Provider::new(provider_type, None));

        // Create the regret node
        let mut regret_node = RegretNode::new(1.0, 0.01, Vec::new(), provider, None);

        // Call the solve method of the regret node
        regret_node.solve(num_iterations, do_randomize_children);
    }
}
