use super::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::neural::nn::shape::ActivationData;
use crate::neural::nn::shape::ActivationType;
use crate::neural::nn::shape::LayerShape;
use crate::neural::nn::shape::LayerType;
use crate::neural::nn::shape::NeuralNetworkShape;

use crate::gen::pheno::rng_wrapper::RngWrapper;

pub struct NeuralNetworkMutater<'a> {
    rng: &'a mut dyn RngWrapper,
}

impl<'a> NeuralNetworkMutater<'a> {
    pub fn new(rng: &'a mut dyn RngWrapper) -> Self {
        Self { rng }
    }

    /// Mutates the given neural network shape and returns an annotated shape.
    ///
    /// # Panics
    /// This function will panic if the random number generator does not provide enough values,
    /// or if an invalid random number is generated.
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn mutate_shape(
        &mut self,
        shape: &NeuralNetworkShape,
    ) -> AnnotatedNeuralNetworkShape {
        let mut mutated_shape = AnnotatedNeuralNetworkShape::new(shape);
        let random_number = self
            .rng
            .fetch_uniform(0.0, 3.0, 1)
            .pop_front()
            .expect("Failed to fetch random number")
            .round() as i32;
        match random_number {
            0 => {
                let shape_num_layers: f32 = shape.num_layers() as f32;
                let position: usize =
                    (self.rng.fetch_uniform(0.0, shape_num_layers, 1).pop_front().unwrap())
                        as usize;
                let layers = fetch_added_layers(self.rng, shape, position);
                mutated_shape.change_layer(position, layers[0].clone());
                mutated_shape.add_layer(position + 1, layers[1].clone());
            },
            1 => {
                let shape_num_layers: f32 = shape.num_layers() as f32;
                let position =
                    (self.rng.fetch_uniform(0.0, shape_num_layers, 1).pop_front().unwrap())
                        as usize;
                let layers = fetch_added_layers(self.rng, shape, position);
                mutated_shape.change_layer(position, layers[0].clone());
                mutated_shape.add_layer(position + 1, layers[1].clone());
            },
            2 => {
                if shape.num_layers() == 1 {
                    return mutated_shape;
                }
                let shape_num_layers: f32 = shape.num_layers() as f32;
                let position =
                    (self.rng.fetch_uniform(0.0, shape_num_layers, 1).pop_front().unwrap())
                        as usize;
                let shape_len = shape.num_layers() - 1;
                if position == 0 {
                    let input_size = shape.get_layer(0).input_size();
                    mutated_shape.remove_layer(0);
                    let layer = mutated_shape.get_layer(0);
                    let new_layer = LayerShape {
                        layer_type: LayerType::Dense {
                            input_size,
                            output_size: layer.output_size(),
                        },
                        activation: layer.activation.clone(),
                    };
                    mutated_shape.change_layer(0, new_layer);
                } else if position == shape_len {
                    let output_size = shape.get_layer(position).output_size();
                    mutated_shape.remove_layer(position);
                    let layer = mutated_shape.get_layer(position - 1);
                    let new_layer = LayerShape {
                        layer_type: LayerType::Dense {
                            input_size: layer.input_size(),
                            output_size,
                        },
                        activation: layer.activation.clone(),
                    };
                    mutated_shape.change_layer(position - 1, new_layer);
                } else {
                    mutated_shape.remove_layer(position);
                    let layer = mutated_shape.get_layer(position);
                    let new_layer = LayerShape {
                        layer_type: LayerType::Dense {
                            input_size: mutated_shape.get_layer(position - 1).output_size(),
                            output_size: layer.output_size(),
                        },
                        activation: layer.activation.clone(),
                    };
                    mutated_shape.change_layer(position, new_layer);
                }
            },
            _ => {
                panic!("Invalid random number generated");
            },
        }
        mutated_shape
    }
}

/// Fetches a random activation data using the provided RNG.
///
/// # Panics
/// This function will panic if the random number generator does not provide enough values,
/// or if an invalid random number is generated.
pub fn fetch_activation_data(rng: &mut dyn RngWrapper) -> ActivationData {
    let random_number: i32 = rng.fetch_uniform(0.0, 4.0, 1).pop_front().unwrap() as i32;
    match random_number {
        0 => ActivationData::new(ActivationType::ReLU),
        1 => ActivationData::new(ActivationType::Sigmoid),
        2 => ActivationData::new(ActivationType::Tanh),
        3 => {
            let random_temperature: f64 =
                rng.fetch_uniform(0.0, 5.0, 1).pop_front().unwrap() as f64;
            ActivationData::new_softmax(random_temperature)
        },
        _ => panic!("Invalid random number generated"),
    }
}

#[allow(clippy::needless_late_init)]
fn fetch_added_layers(
    rng: &mut dyn RngWrapper,
    shape: &NeuralNetworkShape,
    position: usize,
) -> Vec<LayerShape> {
    let activation = fetch_activation_data(rng);

    let random_number: usize = rng.fetch_uniform(0.0, 3.0, 1).pop_front().unwrap() as usize;

    let layer = shape.get_layer(position);
    let layer_input_size: f32 = layer.input_size() as f32;
    let closest_power_of_two = (layer_input_size).log2().floor().exp2();

    let begin_size = layer.input_size();
    let end_size = layer.output_size();

    let mut inner_size: usize = match random_number {
        1 => (closest_power_of_two / 2.0) as usize,
        2 => (closest_power_of_two * 2.0) as usize,
        _ => closest_power_of_two as usize,
    };

    // cap the inner size at an acceptable value
    if inner_size > 1024 {
        inner_size = 1024;
    }

    let first_layer = LayerShape {
        layer_type: LayerType::Dense { input_size: begin_size, output_size: inner_size },
        activation: activation.clone(),
    };

    let second_layer = LayerShape {
        layer_type: LayerType::Dense { input_size: inner_size, output_size: end_size },
        activation,
    };
    vec![first_layer, second_layer]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::pheno::annotated_nn_shape::LayerChangeType;
    use crate::gen::pheno::rng_wrapper::FakeRng;

    // --------------------------------------------------------------------------------------------------------
    // Test adding a layer
    // Test one layer network

    #[test]
    fn test_one_layer_network_adds_medium_layer_and_relu() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 0.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_one_layer_network_adds_medium_layer_and_sigmoid() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 1.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::Sigmoid)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::Sigmoid)
        );
    }

    #[test]
    fn test_one_layer_network_adds_medium_layer_and_tanh() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 2.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::Tanh)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::Tanh)
        );
    }

    // --------------------------------------------------------------------------------------------------------
    // Test medium sized middle layer

    #[test]
    fn test_one_layer_network_adds_small_layer_and_relu() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 0.0, 1.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 64);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 64);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_one_layer_network_adds_small_layer_and_sigmoid() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 1.0, 1.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 64);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::Sigmoid)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 64);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::Sigmoid)
        );
    }

    #[test]
    fn test_one_layer_network_adds_small_layer_and_tanh() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 2.0, 1.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 64);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::Tanh)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 64);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::Tanh)
        );
    }

    // --------------------------------------------------------------------------------------------------------
    // Test medium sized middle layer

    #[test]
    fn test_one_layer_network_adds_big_layer_and_relu() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 0.0, 2.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 256);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 256);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_one_layer_network_adds_big_layer_and_sigmoid() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 1.0, 2.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 256);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::Sigmoid)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 256);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::Sigmoid)
        );
    }

    #[test]
    fn test_one_layer_network_adds_big_layer_and_tanh() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 0.0, 2.0, 2.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 256);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::Tanh)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 256);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::Tanh)
        );
    }

    // --------------------------------------------------------------------------------------------------------
    // Test two layer network

    #[test]
    fn test_two_layer_network_adds_medium_layer_and_relu() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 1.0, 0.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 3);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(2).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(2).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(2).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(2).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_two_layer_network_adds_small_layer_and_relu() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 1.0, 0.0, 1.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 3);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 64);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(2).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(2).input_size(), 64);
        assert_eq!(mutated_shape.get_layer(2).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(2).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_two_layer_network_adds_big_layer_and_relu() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        // The third random number should pick the activation type of the new layer
        // The fourth random number should pick the size of the new layer
        let mut rng = FakeRng::new(vec![0.0, 1.0, 0.0, 2.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 3);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 256);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(2).change_type, LayerChangeType::Add);
        assert_eq!(mutated_shape.get_layer(2).input_size(), 256);
        assert_eq!(mutated_shape.get_layer(2).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(2).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    // --------------------------------------------------------------------------------------------------------
    // Test removing a layer
    // Test one layer network

    #[test]
    fn test_one_layer_network_does_not_remove() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        let mut rng = FakeRng::new(vec![2.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![LayerShape {
                layer_type: LayerType::Dense { input_size: 196, output_size: 10 },
                activation: ActivationData::new(ActivationType::ReLU),
            }],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 1);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    // --------------------------------------------------------------------------------------------------------
    // Test removing a layer
    // Test two layer network

    #[test]
    fn test_two_layer_network_remove_first_leaves_correct_dimensions() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        let mut rng = FakeRng::new(vec![2.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 1);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_two_layer_network_remove_second_leaves_correct_dimensions() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        let mut rng = FakeRng::new(vec![2.0, 1.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);
        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 1);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    // --------------------------------------------------------------------------------------------------------
    // Test removing a layer
    // Test three layer network

    #[test]
    fn test_three_layer_network_remove_first_layer_leaves_correct_dimensions() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        let mut rng = FakeRng::new(vec![2.0, 0.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 64 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 64, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);

        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 64);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 64);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_three_layer_network_remove_middle_layer_leaves_correct_dimensions() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        let mut rng = FakeRng::new(vec![2.0, 1.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 64 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 64, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);

        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }

    #[test]
    fn test_three_layer_network_remove_last_layer_leaves_correct_dimensions() {
        // The first random number should pick the type of change to be applied
        // The second random number should pick the position of the change
        let mut rng = FakeRng::new(vec![2.0, 2.0]);
        let shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 196, output_size: 128 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 128, output_size: 64 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
                LayerShape {
                    layer_type: LayerType::Dense { input_size: 64, output_size: 10 },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        };
        let mut mutater = NeuralNetworkMutater::new(&mut rng);
        let mutated_shape = mutater.mutate_shape(&shape);

        assert_eq!(mutated_shape.to_neural_network_shape().num_layers(), 2);
        assert_eq!(mutated_shape.get_annotated_layer(0).change_type, LayerChangeType::None);
        assert_eq!(mutated_shape.get_layer(0).input_size(), 196);
        assert_eq!(mutated_shape.get_layer(0).output_size(), 128);
        assert_eq!(
            mutated_shape.get_layer(0).activation,
            ActivationData::new(ActivationType::ReLU)
        );
        assert_eq!(mutated_shape.get_annotated_layer(1).change_type, LayerChangeType::Change);
        assert_eq!(mutated_shape.get_layer(1).input_size(), 128);
        assert_eq!(mutated_shape.get_layer(1).output_size(), 10);
        assert_eq!(
            mutated_shape.get_layer(1).activation,
            ActivationData::new(ActivationType::ReLU)
        );
    }
}
