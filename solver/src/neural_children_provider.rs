//! # Neural Children Provider
//!
//! This module provides a children provider for neural network-based regret minimization.

use neural::layer::dense_layer::MatrixParams;
use neural::nn::shape::ActivationData;
use neural::nn::shape::ActivationType;
use neural::nn::shape::AnnotatedNeuralNetworkShape;
use neural::nn::shape::LayerShape;
use neural::nn::shape::LayerType;
use neural::nn::shape::NeuralNetworkShape;
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
    /// params for expected value provider
    num_iterations: usize,
    training_params: TrainingParams,
    all_inputs: Vec<Vec<f64>>,
    all_targets: Vec<Vec<f64>>,
    utils: WrappedUtils,
}

impl NeuralChildrenProvider {
    /// Creates a new neural children provider with the given neural network shape.
    #[must_use]
    pub const fn new(
        shape: NeuralNetworkShape,
        num_iterations: usize,
        training_params: TrainingParams,
        all_inputs: Vec<Vec<f64>>,
        all_targets: Vec<Vec<f64>>,
        utils: WrappedUtils,
    ) -> Self {
        Self { shape, num_iterations, training_params, all_inputs, all_targets, utils }
    }
}

impl ChildrenProvider<NeuralUserData> for NeuralChildrenProvider {
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        parents_data: Vec<WrappedDecision<NeuralUserData>>,
    ) -> Vec<WrappedRegret<NeuralUserData>> {
        let children_shapes = deduce_children_shapes(&self.shape);
        let num_children = children_shapes.len();
        children_shapes
            .into_iter()
            .map(|shape| {
                let child_state = NeuralState::new(shape);
                let child_data = WrappedDecision::new(NeuralUserData::new(child_state));
                let expected_value_provider = NeuralExpectedValueProvider::new(
                    self.num_iterations,
                    self.training_params.clone(),
                    self.all_inputs.clone(),
                    self.all_targets.clone(),
                    self.utils.clone(),
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

fn deduce_children_shapes(shape: &NeuralNetworkShape) -> Vec<NeuralNetworkShape> {
    let last_layer = shape.layers.last().unwrap();
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
