use super::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::evol::rng::RandomNumberGenerator;
use crate::neural::nn::shape::ActivationType;
use crate::neural::nn::shape::LayerShape;
use crate::neural::nn::shape::LayerType;
use crate::neural::nn::shape::NeuralNetworkShape;

pub struct NeuralNetworkMutater<'a> {
    rng: &'a mut RandomNumberGenerator,
}

impl<'a> NeuralNetworkMutater<'a> {
    pub fn new(rng: &'a mut RandomNumberGenerator) -> Self {
        Self { rng }
    }

    pub fn mutate_shape(&mut self, shape: NeuralNetworkShape) -> AnnotatedNeuralNetworkShape {
        let mut mutated_shape = AnnotatedNeuralNetworkShape::new(shape.clone());
        let random_number = self.rng.fetch_uniform(0.0, 3.0, 1).pop_front().unwrap() as i32;
        match random_number {
            0 => {
                let position = self
                    .rng
                    .fetch_uniform(0.0, shape.num_layers() as f32, 1)
                    .pop_front()
                    .unwrap() as usize;
                let layers = fetch_added_layers(self.rng, &shape, position);
                mutated_shape.add_layer(position, layers[0].clone());
                mutated_shape.add_layer(position + 1, layers[1].clone());
            }
            1 => {
                let activation = fetch_activation_type(self.rng);
                let position = self
                    .rng
                    .fetch_uniform(0.0, shape.num_layers() as f32, 1)
                    .pop_front()
                    .unwrap() as usize;
                let mut layer = mutated_shape.get_layer(position).clone();
                layer.activation = activation;
                mutated_shape.change_layer(position, layer);
            }
            2 => {
                let position = self
                    .rng
                    .fetch_uniform(0.0, shape.num_layers() as f32, 1)
                    .pop_front()
                    .unwrap() as usize;
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
                        activation: layer.activation,
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
                        activation: layer.activation,
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
                        activation: layer.activation,
                    };
                    mutated_shape.change_layer(position, new_layer);
                }
            }
            _ => {
                panic!("Invalid random number generated");
            }
        }
        mutated_shape
    }
}

fn fetch_activation_type(rng: &mut RandomNumberGenerator) -> ActivationType {
    let random_number = rng.fetch_uniform(0.0, 3.0, 1).pop_front().unwrap() as i32;
    match random_number {
        0 => ActivationType::ReLU,
        1 => ActivationType::Sigmoid,
        2 => ActivationType::Tanh,
        _ => panic!("Invalid random number generated"),
    }
}

#[allow(clippy::needless_late_init)]
fn fetch_added_layers(
    rng: &mut RandomNumberGenerator,
    shape: &NeuralNetworkShape,
    position: usize,
) -> Vec<LayerShape> {
    let activation = fetch_activation_type(rng);
    let inner_size = rng.fetch_uniform(1.0, 1024.0, 1).pop_front().unwrap() as usize;

    let begin_size;
    let end_size;
    if position == 0 {
        begin_size = shape.get_layer(0).input_size();
        end_size = shape.get_layer(0).input_size();
    } else if position == shape.num_layers() - 1 {
        begin_size = shape.get_layer(position).output_size();
        end_size = shape.get_layer(position).output_size();
    } else {
        begin_size = shape.get_layer(position - 1).output_size();
        end_size = shape.get_layer(position).input_size();
    }

    let first_layer = LayerShape {
        layer_type: LayerType::Dense {
            input_size: begin_size,
            output_size: inner_size,
        },
        activation,
    };

    let second_layer = LayerShape {
        layer_type: LayerType::Dense {
            input_size: inner_size,
            output_size: end_size,
        },
        activation: ActivationType::ReLU,
    };
    vec![first_layer, second_layer]
}
