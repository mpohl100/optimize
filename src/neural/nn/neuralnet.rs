use crate::gen::pheno::annotated_nn_shape::{AnnotatedNeuralNetworkShape, LayerChangeType};
use crate::neural::activation::{
    activate::ActivationTrait, relu::ReLU, sigmoid::Sigmoid, tanh::Tanh,
};
use crate::neural::layer::dense_layer::DenseLayer;
use crate::neural::layer::Layer;
use crate::neural::nn::shape::*;

use std::boxed::Box;

/// A neural network.
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Box<dyn ActivationTrait>>,
    shape: NeuralNetworkShape,
}

impl NeuralNetwork {
    /// Creates a new `NeuralNetwork` from the given shape.
    pub fn new(shape: NeuralNetworkShape) -> Self {
        let shape_clone = shape.clone();
        let mut network = NeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape,
        };

        // Initialize layers and activations based on the provided shape.
        for layer_shape in shape_clone.layers {
            // Here you would instantiate the appropriate Layer and Activation objects.
            let layer = Box::new(DenseLayer::new(
                layer_shape.input_size(),
                layer_shape.output_size(),
            ));
            let activation = match layer_shape.activation {
                ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait>,
            };

            network.add_activation_and_layer(activation, layer);
        }

        network
    }

    /// Creates a new `NeuralNetwork` from the given model directory.
    pub fn from_disk(model_directory: &String) -> NeuralNetwork {
        let shape = NeuralNetworkShape::from_disk(model_directory);
        let mut network = NeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: shape.clone(),
        };

        for i in 0..shape.layers.len() {
            let layer = match &shape.layers[i].layer_type() {
                LayerType::Dense {
                    input_size,
                    output_size,
                } => {
                    let mut layer = DenseLayer::new(*input_size, *output_size);
                    layer
                        .read(&format!("{}/layers/layer_{}", model_directory, i))
                        .unwrap();
                    Box::new(layer) as Box<dyn Layer>
                }
            };
            let activation = match shape.layers[i].activation {
                ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait>,
            };

            network.add_activation_and_layer(activation, layer);
        }

        network
    }

    /// Adds an activation and a layer to the neural network.
    fn add_activation_and_layer(
        &mut self,
        activation: Box<dyn ActivationTrait>,
        layer: Box<dyn Layer>,
    ) {
        self.activations.push(activation);
        self.layers.push(layer);
    }

    /// Performs a forward pass through the network with the given input.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&self.activations) {
            output = layer.forward(&output);
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a backward pass through the network with the given output gradient.
    pub fn backward(&mut self, grad_output: Vec<f64>) {
        let mut grad = grad_output;
        for (layer, activation) in self
            .layers
            .iter_mut()
            .rev()
            .zip(self.activations.iter_mut().rev())
        {
            grad = activation.backward(&grad);
            grad = layer.backward(&grad);
        }
    }

    /// Trains the neural network using the given inputs, targets, learning rate, and number of epochs.
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            print!("Epoch: {}\r", epochs);
            for (input, target) in inputs.iter().zip(targets) {
                // Forward pass
                let output = self.forward(input.as_slice());

                // Calculate loss gradient (e.g., mean squared error)
                let grad_output: Vec<f64> = output.iter().zip(target).map(|(o, t)| o - t).collect();

                // Backward pass
                self.backward(grad_output);

                // Update weights
                for layer in &mut self.layers {
                    layer.update_weights(learning_rate);
                }
            }
        }
    }

    /// Makes a prediction based on a single input by performing a forward pass.
    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input.as_slice())
    }

    /// Returns the input size of the first layer in the network.
    pub fn input_size(&self) -> usize {
        self.shape
            .layers
            .first()
            .map_or(0, |layer| layer.input_size())
    }

    /// Returns the output size of the last layer in the network.
    pub fn output_size(&self) -> usize {
        self.shape
            .layers
            .last()
            .map_or(0, |layer| layer.output_size())
    }

    pub fn shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    pub fn save_layers(&self, model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        // make a layers subdirectory
        std::fs::create_dir_all(format!("{}/layers", model_directory))?;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.save(&format!("{}/layers/layer_{}", model_directory, i))?;
        }
        Ok(())
    }

    pub fn save(&self, model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        // remove the directory if it exists
        let backup_directory = format!("{}_backup", model_directory);
        if std::fs::metadata(&model_directory).is_ok() {
            // copy the directory to a backup
            std::fs::rename(&model_directory, &backup_directory)?;
            std::fs::create_dir_all(&model_directory)?;
        } else {
            // create directory if it doesn't exist
            std::fs::create_dir_all(&model_directory)?;
        }

        let shape = self.shape();
        shape.to_yaml(model_directory.clone());
        self.save_layers(model_directory)?;

        // if backup directory exists, remove it
        if std::fs::metadata(&backup_directory).is_ok() {
            std::fs::remove_dir_all(&backup_directory)?;
        }

        Ok(())
    }

    pub fn adapt_to_shape(&mut self, shape: AnnotatedNeuralNetworkShape) {
        for (i, layer) in shape.layers.iter().enumerate() {
            match layer.change_type {
                LayerChangeType::Add => {
                    let new_layer = match &layer.layer.layer_type() {
                        LayerType::Dense {
                            input_size,
                            output_size,
                        } => Box::new(DenseLayer::new(*input_size, *output_size)) as Box<dyn Layer>,
                    };
                    let activation = match layer.layer.activation {
                        ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait>,
                        ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait>,
                        ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait>,
                    };
                    self.add_activation_and_layer_at_position(i, activation, new_layer);
                }
                LayerChangeType::Remove => {
                    self.layers.remove(i);
                    self.activations.remove(i);
                }
                LayerChangeType::Change => {
                    let mut changed = false;
                    match &layer.layer.layer_type() {
                        LayerType::Dense {
                            input_size,
                            output_size,
                        } => {
                            if *input_size != self.layers[i].input_size()
                                || *output_size != self.layers[i].output_size()
                            {
                                self.layers[i].resize(*input_size, *output_size);
                                changed = true;
                            }
                        }
                    };
                    if changed {
                        continue;
                    }
                    let activation = match layer.layer.activation {
                        ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait>,
                        ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait>,
                        ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait>,
                    };
                    self.activations[i] = activation;
                }
                LayerChangeType::None => {}
            }
        }
        self.shape = shape.to_neural_network_shape();
    }

    fn add_activation_and_layer_at_position(
        &mut self,
        position: usize,
        activation: Box<dyn ActivationTrait>,
        layer: Box<dyn Layer>,
    ) {
        self.activations.insert(position, activation);
        self.layers.insert(position, layer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::nn::shape::{ActivationType, LayerShape};

    #[test]
    fn test_neural_network_train() {
        let mut nn = NeuralNetwork::new(NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 3,
                        output_size: 3,
                    },
                    activation: ActivationType::Sigmoid,
                },
                LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 3,
                        output_size: 3,
                    },
                    activation: ActivationType::ReLU,
                },
            ],
        });

        let inputs = vec![vec![1.0, 1.0, 1.0]];
        let targets = vec![vec![0.0, 0.0, 0.0]];

        nn.train(&inputs, &targets, 0.01, 100);

        let prediction = nn.predict(inputs[0].clone());
        // print targets[0]
        println!("{:?}", targets[0]);
        // print prediction
        println!("{:?}", prediction);
        assert_eq!(prediction.len(), 3);
        // assert that the prediction is close to the target
        for (p, t) in prediction.iter().zip(&targets[0]) {
            assert!((p - t).abs() < 1e-4);
        }
    }
}
