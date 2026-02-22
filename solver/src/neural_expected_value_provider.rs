//! # Neural Expected Value Provider
//!
//! This module provides an expected value provider for neural network-based regret minimization.

use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_data::RandomTrainingDataView;
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;
use num_traits::NumCast;
use regret::provider::ExpectedValueProvider;
use regret::user_data::WrappedDecision;
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

use crate::neural_children_provider::NeuralUserData;

/// Expected value provider for neural network-based decisions.
#[derive(Debug, Clone)]
pub struct NeuralExpectedValueProvider {
    /// Counter to track how many times `get_expected_value` was called.
    counter: Arc<Mutex<usize>>,
    /// Number of iterations to split training data.
    num_iterations: usize,
    /// Training parameters.
    training_params: TrainingParams,
    /// Randomly shuffled view of the training data.
    random_training_data_view: RandomTrainingDataView,
}

/// Data importer for a subset of training data.
#[derive(Clone)]
struct SubsetDataImporter {
    data: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
}

impl DataImporter for SubsetDataImporter {
    fn get_data(&self) -> SessionData {
        SessionData { data: self.data.clone(), labels: self.labels.clone() }
    }
}

impl NeuralExpectedValueProvider {
    /// Creates a new neural expected value provider.
    #[must_use]
    pub fn new(
        num_iterations: usize,
        training_params: TrainingParams,
        random_training_data_view: RandomTrainingDataView,
    ) -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            num_iterations,
            training_params,
            random_training_data_view,
        }
    }
}

impl ExpectedValueProvider<NeuralUserData> for NeuralExpectedValueProvider {
    /// Returns the expected value for the given parent data.
    fn get_expected_value(
        &self,
        parents_data: Vec<WrappedDecision<NeuralUserData>>,
    ) -> f64 {
        // Get and increment counter
        let current_counter = {
            let mut counter = safe_lock(&self.counter);
            let current = *counter;
            *counter += 1;
            current
        };

        // Extract the neural network from the first parent's state
        let neural_network = if let Some(parent) = parents_data.first() {
            let user_data = parent.get_decision_data();
            let state = user_data.get_state();
            if let Some(nn) = state.get_neural_network() {
                nn.clone()
            } else {
                // No neural network in state, return 0.0
                return 0.0;
            }
        } else {
            // No parent data, return 0.0
            return 0.0;
        };

        // Derive `from` and `to` fractions from the counter and num_iterations
        let num_iterations_float: f64 = NumCast::from(self.num_iterations).unwrap_or(1.0);
        let current_float: f64 = NumCast::from(current_counter).unwrap_or(0.0);
        let from = current_float / num_iterations_float;
        let to = ((current_float + 1.0) / num_iterations_float).min(1.0);

        // If we're beyond the available data range, return 0.0
        if from >= 1.0 {
            return 0.0;
        }

        // Retrieve the random subset for this iteration
        let subset = self.random_training_data_view.get_samples_and_labels(from, to);
        let inputs = subset.inputs();
        let targets = subset.targets();

        // Create a simple data importer for this subset
        let data_importer = Box::new(SubsetDataImporter { data: inputs, labels: targets });

        // Create a modified training params with validation_split set to 1.0
        let mut modified_params = self.training_params.clone();
        modified_params.set_validation_split(1.0);

        // Create a new TrainingSession from the neural network
        let Ok(mut training_session) =
            TrainingSession::new_from_nn(neural_network, modified_params, data_importer)
        else {
            return 0.0;
        };

        // Train and return accuracy
        training_session.train().unwrap_or(0.0)
    }
}
