//! # Neural Solver
//!
//! This module provides a solver that combines neural networks with regret minimization.

use neural::nn::nn_factory::{new_trainable_neural_network, NeuralNetworkCreationArguments};
use neural::nn::nn_trait::WrappedNeuralNetwork;
use neural::nn::shape::NeuralNetworkShape;
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_data::WrappedTrainingData;
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;
use neural::utilities::util::WrappedUtils;
use regret::provider::{Provider, ProviderType, WrappedChildrenProvider, WrappedProvider};
use regret::regret_node::RegretNode;

use crate::neural_children_provider::NeuralChildrenProvider;

/// A solver that uses neural networks and regret minimization.
#[derive(Clone)]
pub struct NeuralSolver {
    /// The neural network shape to use for solving.
    shape: NeuralNetworkShape,
    training_params: TrainingParams,
    all_inputs: Vec<Vec<f64>>,
    all_targets: Vec<Vec<f64>>,
    utils: WrappedUtils,
}

/// Data importer that provides the full dataset.
#[derive(Clone)]
struct AllDataImporter {
    data: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
}

impl DataImporter for AllDataImporter {
    fn get_data(&self) -> SessionData {
        SessionData { data: self.data.clone(), labels: self.labels.clone() }
    }
}

impl NeuralSolver {
    /// Creates a new neural solver with the given neural network shape.
    #[must_use]
    pub const fn new(
        shape: NeuralNetworkShape,
        training_params: TrainingParams,
        all_inputs: Vec<Vec<f64>>,
        all_targets: Vec<Vec<f64>>,
        utils: WrappedUtils,
    ) -> Self {
        Self { shape, training_params, all_inputs, all_targets, utils }
    }

    /// Solves the problem using regret minimization with the neural network.
    ///
    /// Returns the `WrappedNeuralNetwork` from the child state that achieved the
    /// highest average expected value (percentage) during regret minimization.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - The number of iterations to run the regret minimization algorithm.
    /// * `do_randomize_children` - Whether to randomize children during solving.
    /// * `min_accuracy` - The minimum accuracy threshold. The solver loops, using the winner's
    ///   shape as the starting shape for the next loop, until the winner's accuracy reaches this
    ///   threshold. Returns `None` if accuracy decreases between loop runs.
    pub fn solve(
        &mut self,
        num_iterations: usize,
        do_randomize_children: bool,
        min_accuracy: f64,
    ) -> Option<WrappedNeuralNetwork> {
        let mut current_shape = self.shape.clone();
        let mut prev_accuracy: f64 = -1.0;
        loop {
            // Create the children provider with the current shape
            let wrapped_training_data =
                WrappedTrainingData::new(self.all_inputs.clone(), self.all_targets.clone());
            let children_provider = NeuralChildrenProvider::new(
                current_shape.clone(),
                self.training_params.clone(),
                wrapped_training_data,
                self.utils.clone(),
            );

            // Create the wrapped provider with the children provider
            let provider = WrappedProvider::new(Provider::new(
                ProviderType::Children(WrappedChildrenProvider::new(Box::new(children_provider))),
                None,
            ));

            // Create the regret node with fixed probability
            let mut node = RegretNode::new(1.0, 0.01, vec![], provider, Some(1.0));

            // Solve using the regret minimization algorithm
            node.solve(num_iterations, do_randomize_children);

            // Find the child with the highest average expected value
            let children = node.get_children();
            let best_child = children.iter().max_by(|a, b| {
                a.get_average_expected_value()
                    .partial_cmp(&b.get_average_expected_value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;

            let winner_accuracy = best_child.get_average_expected_value();

            // Get the best shape from the child's NeuralState
            let best_shape =
                best_child.get_user_data()?.get_decision_data().get_state().get_shape().clone();

            // If accuracy decreased compared to the previous loop run, the algorithm failed
            if winner_accuracy < prev_accuracy {
                return None;
            }

            // If the winner meets or exceeds the minimum accuracy, train a final model and return
            if winner_accuracy >= min_accuracy {
                // Create a final training session using the best shape on the full dataset
                let mut final_params = self.training_params.clone();
                final_params.set_shape(best_shape);

                let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
                    final_params.shape().clone(),
                    final_params.levels(),
                    final_params.pre_shape(),
                    "neural_solver_final".to_string(),
                    self.utils.clone(),
                ));

                let data_importer: Box<dyn DataImporter> = Box::new(AllDataImporter {
                    data: self.all_inputs.clone(),
                    labels: self.all_targets.clone(),
                });

                let Ok(mut training_session) =
                    TrainingSession::new_from_nn(nn, final_params, data_importer)
                else {
                    return None;
                };

                training_session.train().ok()?;
                return Some(training_session.get_nn().to_neural_network());
            }

            // Update for next loop: use winner's shape as the new starting shape
            prev_accuracy = winner_accuracy;
            current_shape = best_shape;
        }
    }
}
