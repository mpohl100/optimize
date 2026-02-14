//! # Neural Solver
//!
//! This module provides a solver that combines neural networks with regret minimization.

use neural::nn::shape::NeuralNetworkShape;
use regret::provider::{Provider, ProviderType, WrappedChildrenProvider, WrappedProvider};
use regret::regret_node::RegretNode;

use crate::neural_children_provider::NeuralChildrenProvider;

/// A solver that uses neural networks and regret minimization.
#[derive(Clone)]
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
        &mut self,
        num_iterations: usize,
        do_randomize_children: bool,
    ) {
        // Create the children provider with the neural network shape
        let children_provider = NeuralChildrenProvider::new(self.shape.clone());

        // Create the wrapped provider with the children provider
        let provider = WrappedProvider::new(Provider::new(
            ProviderType::Children(WrappedChildrenProvider::new(Box::new(children_provider))),
            None,
        ));

        // Create the regret node with fixed probability
        let mut node = RegretNode::new(1.0, 0.01, vec![], provider, Some(1.0));

        // Solve using the regret minimization algorithm
        node.solve(num_iterations, do_randomize_children);
    }
}
