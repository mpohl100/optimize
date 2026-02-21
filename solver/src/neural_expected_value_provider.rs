//! # Neural Expected Value Provider
//!
//! This module provides an expected value provider for neural network-based regret minimization.

use neural::nn::nn_factory::{new_trainable_neural_network, NeuralNetworkCreationArguments};
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;
use neural::utilities::util::WrappedUtils;
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
    /// All training data inputs.
    all_inputs: Arc<Vec<Vec<f64>>>,
    /// All training data targets.
    all_targets: Arc<Vec<Vec<f64>>>,
    /// Utilities for neural network creation.
    utils: WrappedUtils,
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
        all_inputs: Vec<Vec<f64>>,
        all_targets: Vec<Vec<f64>>,
        utils: WrappedUtils,
    ) -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
            num_iterations,
            training_params,
            all_inputs: Arc::new(all_inputs),
            all_targets: Arc::new(all_targets),
            utils,
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

        // Extract the shape from the first parent's state
        let shape = if let Some(parent) = parents_data.first() {
            let user_data = parent.get_decision_data();
            let state = user_data.get_state();
            state.get_shape().clone()
        } else {
            // No parent data, return 0.0
            return 0.0;
        };

        // Calculate N: total samples divided by num_iterations
        let total_samples = self.all_inputs.len();
        let n = if self.num_iterations > 0 {
            total_samples / self.num_iterations
        } else {
            total_samples
        };

        // Calculate sample range: from counter*N to (counter+1)*N
        let start_idx = current_counter * n;
        let end_idx = ((current_counter + 1) * n).min(total_samples);

        // If we're beyond the available data, return 0.0
        if start_idx >= total_samples {
            return 0.0;
        }

        // Extract the samples for this iteration
        let inputs: Vec<Vec<f64>> = self.all_inputs[start_idx..end_idx].to_vec();
        let targets: Vec<Vec<f64>> = self.all_targets[start_idx..end_idx].to_vec();

        // Create a simple data importer for this subset
        let data_importer: Box<dyn DataImporter> =
            Box::new(SubsetDataImporter { data: inputs, labels: targets });

        // Create modified training params with the child shape and validation_split set to 1.0
        let mut modified_params = self.training_params.clone();
        modified_params.set_shape(shape);
        modified_params.set_validation_split(1.0);

        // Create a new neural network for this child shape
        let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            modified_params.shape().clone(),
            modified_params.levels(),
            modified_params.pre_shape(),
            format!("solver_child_{current_counter}"),
            self.utils.clone(),
        ));

        // Create a new TrainingSession from the neural network
        let Ok(mut training_session) =
            TrainingSession::new_from_nn(nn, modified_params, data_importer)
        else {
            return 0.0;
        };

        // Train and return accuracy
        training_session.train().unwrap_or(0.0)
    }
}
