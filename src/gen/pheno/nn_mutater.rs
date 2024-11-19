use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::nn::shape::ActivationType;
use learn::evol::rng::RandomNumberGenerator;

struct NeuralNetworkMutater{
    rng: &mut RandomNumberGenerator,
}

impl NeuralNetworkMutater{
    pub fn new(rng: &mut RandomNumberGenerator) -> Self {
        Self {
            rng: rng,
        }
    }

    pub fn mutate_shape(&self, shape: NeuralNetworkShape) -> NeuralNetworkShape {
        let mut mutated_shape = shape.clone();
        let mut rng = self.rng;
        let random_number = rng.fetch_uniform(0.0, 3.0, 1).pop_front().unwrap() as i32;
        match random_number {
            0 => {
                let layers = fetch_added_layers(rng, &shape);
                mutated_shape.add_layer(position, layers[0]);
                mutated_shape.add_layer(position + 1, layers[1]);
            },
            1 => {
                let activation = ActivationType::Sigmoid;
                let position = *rng.fetch_uniform(0.0, shape.len() as f32, 1).pop_front().unwrap() as usize;
                let mut layer = mutated_shape.get_layer(position);
                layer.activation = activation;
                mutated_shape.change_layer(position, activation);
            },
            2 => {
                let position = *rng.fetch_uniform(0.0, shape.len() as f32, 1).pop_front().unwrap() as usize;
                let shape_len = shape.len();
                match position {
                    0 => {
                        let input_size = shape.get_layer(0).input_size();
                        mutated_shape.remove_layer(0);
                        let layer = mutated_shape.get_layer(0);
                        let new_layer = LayerShape {
                            layer_type: LayerType::Dense {
                                input_size: input_size,
                                output_size: layer.get_output_size(),
                            },
                            activation: layer.activation,
                        };
                        mutated_shape.change_layer(0, new_layer);
                    },
                    shape_len => {
                        let output_size = shape.get_layer(position).output_size();
                        mutated_shape.remove_layer(position);
                        let layer = mutated_shape.get_layer(position - 1);
                        let new_layer = LayerShape {
                            layer_type: LayerType::Dense {
                                input_size: layer.get_input_size(),
                                output_size: output_size,
                            },
                            activation: layer.activation,
                        };
                        mutated_shape.change_layer(position - 1, new_layer);
                    }
                    _ => {
                        mutated_shape.remove_layer(position);
                        let layer = mutated_shape.get_layer(position);
                        let new_layer = LayerShape {
                            layer_type: LayerType::Dense {
                                input_size: mutated_shape.get_layer(position - 1).get_output_size(),
                                output_size: layer.get_output_size(),
                            },
                            activation: layer.activation,
                        };
                        mutated_shape.change_layer(position, new_layer);
                    }
                }
            },
            _ => {
                panic!("Invalid random number generated");
            }
        }
        mutated_shape
    }
}

fn fetch_activation_type(&mut rng: RandomNumberGenerator) -> ActivationType {
    let random_number = rng.fetch_uniform(0.0, 3.0, 1).pop_front().unwrap() as i32;
    match random_number {
        0 => ActivationType::ReLU,
        1 => ActivationType::Sigmoid,
        2 => ActivationType::Tanh,
        _ => panic!("Invalid random number generated"),
    }
}

fn fetch_added_layers(rng: &mut RandomNumberGenerator, shape: &NeuralNetworkShape) -> Vec<LayerShape> {
    let activation = fetch_activation_type(rng);
    let position = *rng.fetch_uniform(0.0, shape.len() as f32, 1).pop_front().unwrap() as usize;
    let inner_size = *rng.fetch_uniform(1.0, 1024.0, 1).pop_front().unwrap() as usize;

    let begin_size = match position {
        0 => shape.get_layer(0).get_input_size(),
        _ => shape.get_layer(position - 1).get_output_size(),
    };

    let shape_len = shape.len();
    let end_size = match position {
        shape_len => shape.get_layer(shape.len() - 1).get_output_size(),
        _ => shape.get_layer(position).get_input_size(),
    };

    let first_layer = LayerShape {
        layer_type: LayerType::Dense {
            input_size: begin_size,
            output_size: inner_size,
        },
        activation: activation,
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