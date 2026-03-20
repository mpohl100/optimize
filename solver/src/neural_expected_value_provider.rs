//! # Neural Expected Value Provider
//!
//! This module provides an expected value provider for neural network-based regret minimization.

use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_data::RandomTrainingDataView;
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;
use regret::provider::ExpectedValueProvider;
use regret::user_data::WrappedDecision;

use crate::neural_children_provider::NeuralUserData;

/// Expected value provider for neural network-based decisions.
#[derive(Debug, Clone)]
pub struct NeuralExpectedValueProvider {
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
    #[allow(clippy::missing_const_for_fn)]
    #[must_use]
    pub fn new(
        training_params: TrainingParams,
        random_training_data_view: RandomTrainingDataView,
    ) -> Self {
        Self { training_params, random_training_data_view }
    }
}

impl ExpectedValueProvider<NeuralUserData> for NeuralExpectedValueProvider {
    /// Returns the expected value for the given parent data.
    fn get_expected_value(
        &self,
        parents_data: Vec<WrappedDecision<NeuralUserData>>,
    ) -> f64 {
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

        // Always train on the complete randomized training set.
        let subset = self.random_training_data_view.get_samples_and_labels(0.0, 1.0);
        let inputs = subset.inputs();
        let targets = subset.targets();

        // Create a simple data importer for this subset
        let data_importer = Box::new(SubsetDataImporter { data: inputs, labels: targets });

        // Create a new TrainingSession from the neural network
        let Ok(mut training_session) = TrainingSession::new_from_nn(
            neural_network,
            self.training_params.clone(),
            data_importer,
        ) else {
            return 0.0;
        };

        // Train and return accuracy
        training_session.train().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_children_provider::NeuralUserData;
    use crate::neural_state::NeuralState;
    use neural::layer::dense_layer::MatrixParams;
    use neural::nn::nn_factory::{new_trainable_neural_network, NeuralNetworkCreationArguments};
    use neural::nn::shape::{
        ActivationData, ActivationType, LayerShape, LayerType, NeuralNetworkShape,
    };
    use neural::training::training_data::WrappedTrainingData;
    use neural::utilities::util::{Utils, WrappedUtils};
    use regret::provider::ExpectedValueProvider;

    #[test]
    fn uses_passed_validation_split() {
        let shape = NeuralNetworkShape::new(vec![LayerShape {
            layer_type: LayerType::Dense {
                input_size: 2,
                output_size: 1,
                matrix_params: MatrixParams { slice_rows: 10, slice_cols: 10 },
            },
            activation: ActivationData::new(ActivationType::Sigmoid),
        }]);
        let training_params =
            TrainingParams::new(shape.clone(), None, None, 0.5, 0.01, 1, 0.1, 1, false, 0.0);
        let wrapped_training_data = WrappedTrainingData::new(
            vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
        );
        let random_training_data_view = RandomTrainingDataView::new(wrapped_training_data);
        let provider = NeuralExpectedValueProvider::new(training_params, random_training_data_view);
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 1));
        let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            shape.clone(),
            None,
            None,
            "neural_expected_value_provider_test".to_string(),
            utils,
        ));
        let parent = WrappedDecision::new(NeuralUserData::new(NeuralState::new_with_nn(shape, nn)));

        let value = provider.get_expected_value(vec![parent]);

        assert!(value.is_finite(), "expected finite accuracy when validation_split < 1.0");
    }
}
