//! # Neural Children Provider
//!
//! This module provides a children provider for neural network-based regret minimization.

use neural::layer::dense_layer::MatrixParams;
use neural::nn::nn_factory::{new_trainable_neural_network, NeuralNetworkCreationArguments};
use neural::nn::shape::ActivationData;
use neural::nn::shape::ActivationType;
use neural::nn::shape::AnnotatedNeuralNetworkShape;
use neural::nn::shape::LayerShape;
use neural::nn::shape::LayerType;
use neural::nn::shape::NeuralNetworkShape;
use neural::training::training_data::{RandomTrainingDataView, WrappedTrainingData};
use neural::training::training_params::TrainingParams;
use neural::utilities::util::WrappedUtils;
use regret::provider::ChildrenProvider;
use regret::provider::Provider;
use regret::provider::ProviderType;
use regret::provider::WrappedExpectedValueProvider;
use regret::provider::WrappedProvider;
use regret::regret_node::RegretNode;
use regret::regret_node::WrappedRegret;
use regret::user_data::{DecisionTrait, WrappedDecision};

use crate::neural_expected_value_provider::NeuralExpectedValueProvider;
use crate::neural_state::{NeuralState, StateTrait};

use num_traits::NumCast;

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
    shape: NeuralNetworkShape,
    training_params: TrainingParams,
    wrapped_training_data: WrappedTrainingData,
    utils: WrappedUtils,
}

impl NeuralChildrenProvider {
    /// Creates a new neural children provider with the given neural network shape.
    #[must_use]
    pub const fn new(
        shape: NeuralNetworkShape,
        training_params: TrainingParams,
        wrapped_training_data: WrappedTrainingData,
        utils: WrappedUtils,
    ) -> Self {
        Self { shape, training_params, wrapped_training_data, utils }
    }
}

impl ChildrenProvider<NeuralUserData> for NeuralChildrenProvider {
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        parents_data: Vec<WrappedDecision<NeuralUserData>>,
    ) -> Vec<WrappedRegret<NeuralUserData>> {
        let (input_size, output_size) = if self.shape.layers.is_empty() {
            let inputs = self.wrapped_training_data.inputs();
            let targets = self.wrapped_training_data.targets();
            let input_size = inputs.first().map_or(0, Vec::len);
            let output_size = targets.first().map_or(0, Vec::len);
            (input_size, output_size)
        } else {
            let last_layer = self.shape.layers.last().expect("shape has layers");
            (last_layer.input_size(), last_layer.output_size())
        };
        let children_shapes = deduce_children_shapes(&self.shape, input_size, output_size);
        let num_children = children_shapes.len();
        let random_training_data_view =
            RandomTrainingDataView::new(self.wrapped_training_data.clone());
        children_shapes
            .into_iter()
            .enumerate()
            .map(|(idx, shape)| {
                // Build a trainable neural network for this child shape up front
                let mut params = self.training_params.clone();
                params.set_shape(shape.clone());
                let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
                    shape.clone(),
                    params.levels(),
                    params.pre_shape(),
                    format!("solver_child_{idx}"),
                    self.utils.clone(),
                ));
                let child_state = NeuralState::new_with_nn(shape, nn);
                let child_data = WrappedDecision::new(NeuralUserData::new(child_state));
                let expected_value_provider = NeuralExpectedValueProvider::new(
                    self.training_params.clone(),
                    random_training_data_view.clone(),
                );
                let num_children_f64 = NumCast::from(num_children).unwrap_or(1.0);
                let probability = 1.0 / num_children_f64;
                let min_probability = 0.01;
                let wrapped_expected_value_provider =
                    WrappedExpectedValueProvider::new(Box::new(expected_value_provider));
                let provider = Provider::new(
                    ProviderType::ExpectedValue(wrapped_expected_value_provider),
                    Some(child_data),
                );
                let node = RegretNode::new(
                    probability,
                    min_probability,
                    parents_data.clone(),
                    WrappedProvider::new(provider),
                    None,
                );
                WrappedRegret::new(node)
            })
            .collect()
    }
}

fn deduce_children_shapes(
    shape: &NeuralNetworkShape,
    input_size: usize,
    output_size: usize,
) -> Vec<NeuralNetworkShape> {
    if shape.layers.is_empty() {
        return generate_layer_shapes(input_size, output_size)
            .into_iter()
            .map(NeuralNetworkShape::new)
            .collect();
    }
    let last_layer = shape.layers.last().expect("shape has layers");
    let input_size = last_layer.input_size();
    let output_size = last_layer.output_size();
    let annotated_shape = AnnotatedNeuralNetworkShape::new(shape);
    let mut results = Vec::new();
    for layer_shapes in generate_layer_shapes(input_size, output_size) {
        let mut new_shape = annotated_shape.clone();
        new_shape.change_layer(shape.layers.len() - 1, layer_shapes[0].clone());
        new_shape.add_layer(shape.layers.len(), layer_shapes[1].clone());
        let new_shape = new_shape.to_neural_network_shape();
        results.push(new_shape);
    }
    results
}

fn generate_layer_shapes(
    input_size: usize,
    output_size: usize,
) -> Vec<Vec<LayerShape>> {
    let mut results = Vec::new();
    let half_size = (input_size + output_size) / 2;
    let dimensions = [
        [(input_size, input_size), (input_size, output_size)],
        [(input_size, half_size), (half_size, output_size)],
        [(input_size, output_size), (output_size, output_size)],
    ];
    let activations = [
        ActivationData::new(ActivationType::ReLU),
        ActivationData::new(ActivationType::Sigmoid),
        ActivationData::new(ActivationType::Tanh),
        ActivationData::new_softmax(1.0),
        ActivationData::new_softmax(1.5),
        ActivationData::new_softmax(2.0),
    ];
    let matrix_params = MatrixParams { slice_rows: 1000, slice_cols: 1000 };
    for dimension in &dimensions {
        for activation in &activations {
            let layer1 = LayerShape {
                layer_type: LayerType::Dense {
                    input_size: dimension[0].0,
                    output_size: dimension[0].1,
                    matrix_params,
                },
                activation: activation.clone(),
            };
            let layer2 = LayerShape {
                layer_type: LayerType::Dense {
                    input_size: dimension[1].0,
                    output_size: dimension[1].1,
                    matrix_params,
                },
                activation: ActivationData::new(ActivationType::ReLU),
            };
            results.push(vec![layer1, layer2]);
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduce_children_shapes_empty_uses_training_data_sizes() {
        let empty_shape = NeuralNetworkShape::new(vec![]);
        let input_size = 10;
        let output_size = 3;
        let shapes = deduce_children_shapes(&empty_shape, input_size, output_size);
        assert!(!shapes.is_empty(), "should produce children shapes from training data sizes");
        for shape in &shapes {
            assert_eq!(shape.layers.len(), 2, "each generated shape should have 2 layers");
            let first = &shape.layers[0];
            let last = &shape.layers[shape.layers.len() - 1];
            assert_eq!(
                first.input_size(),
                input_size,
                "first layer input size should match sample row length"
            );
            assert_eq!(
                last.output_size(),
                output_size,
                "last layer output size should match target row length"
            );
        }
    }

    #[test]
    fn test_deduce_children_shapes_nonempty_uses_last_layer() {
        use neural::layer::dense_layer::MatrixParams;
        use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
        let layers = vec![LayerShape {
            layer_type: LayerType::Dense {
                input_size: 8,
                output_size: 4,
                matrix_params: MatrixParams { slice_rows: 1000, slice_cols: 1000 },
            },
            activation: ActivationData::new(ActivationType::ReLU),
        }];
        let shape = NeuralNetworkShape::new(layers);
        let shapes = deduce_children_shapes(&shape, 0, 0);
        assert!(!shapes.is_empty(), "should produce children shapes from last layer");
        for child_shape in &shapes {
            assert!(
                child_shape.layers.len() >= 2,
                "each child shape should have at least 2 layers"
            );
            let second_to_last = &child_shape.layers[child_shape.layers.len() - 2];
            assert_eq!(
                second_to_last.input_size(),
                8,
                "child shapes should use input_size from last layer"
            );
            let last = child_shape.layers.last().expect("child shape has layers");
            assert_eq!(
                last.output_size(),
                4,
                "child shapes should use output_size from last layer"
            );
        }
    }
}
