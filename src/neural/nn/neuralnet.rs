use crate::gen::pheno::annotated_nn_shape::{AnnotatedNeuralNetworkShape, LayerChangeType};
use crate::neural::activation::{
    activate::ActivationTrait, relu::ReLU, sigmoid::Sigmoid, tanh::Tanh,
};
use crate::neural::layer::dense_layer::DenseLayer;
use crate::neural::layer::Layer;
use crate::neural::nn::shape::*;

use std::boxed::Box;

/// A neural network.
#[derive(Debug, Clone, Default)]
pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer + Send>>,
    activations: Vec<Box<dyn ActivationTrait + Send>>,
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
                ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
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
                    Box::new(layer) as Box<dyn Layer + Send>
                }
            };
            let activation = match shape.layers[i].activation {
                ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
            };

            network.add_activation_and_layer(activation, layer);
        }

        network
    }

    /// Adds an activation and a layer to the neural network.
    fn add_activation_and_layer(
        &mut self,
        activation: Box<dyn ActivationTrait + Send>,
        layer: Box<dyn Layer + Send>,
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
            println!("Epoch: {}\r", epochs);
            let mut loss = 0.0;
            for (input, target) in inputs.iter().zip(targets) {
                // Forward pass
                let output = self.forward(input.as_slice());

                // Calculate loss gradient (e.g., mean squared error)
                // let grad_output: Vec<f64> = output.iter().zip(target).map(|(o, t)| o - t).collect();
                let mut grad_output = Vec::new();
                for i in 0..output.len() {
                    let error = output[i] - target[i];
                    grad_output.push(2.0 * error);
                    loss += error * error;
                }
                // Backward pass
                self.backward(grad_output);

                // Update weights
                for layer in &mut self.layers {
                    layer.update_weights(learning_rate);
                }
            }
            loss /= inputs.len() as f64;
            println!("Loss: {}\r", loss);
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
                        } => Box::new(DenseLayer::new(*input_size, *output_size))
                            as Box<dyn Layer + Send>,
                    };
                    let activation = match layer.layer.activation {
                        ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait + Send>,
                        ActivationType::Sigmoid => {
                            Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>
                        }
                        ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
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
                        ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait + Send>,
                        ActivationType::Sigmoid => {
                            Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>
                        }
                        ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                    };
                    self.activations[i] = activation;
                }
                LayerChangeType::None => {}
            }
        }
        self.shape = shape.to_neural_network_shape();
    }

    pub fn merge(&self, other: NeuralNetwork) -> NeuralNetwork {
        let mut new_nn = NeuralNetwork::default();
        for i in 0..self.layers.len() {
            new_nn.add_activation_and_layer(self.activations[i].clone(), self.layers[i].clone());
        }

        let merge_layer_input_size = self.layers.last().unwrap().output_size();
        let merge_layer_output_size = other.layers.first().unwrap().input_size();
        let merge_layer = Box::new(DenseLayer::new(
            merge_layer_input_size,
            merge_layer_output_size,
        ));
        let merge_activation = Box::new(ReLU);
        new_nn.add_activation_and_layer(merge_activation, merge_layer);

        for i in 0..other.layers.len() {
            new_nn.add_activation_and_layer(other.activations[i].clone(), other.layers[i].clone());
        }
        new_nn.deduce_shape();
        new_nn
    }

    /// Deduces the shape of the neural network based on the layers and activations.
    fn deduce_shape(&mut self) {
        let mut layers = Vec::new();
        for i in 0..self.layers.len() {
            let layer_shape = LayerShape {
                layer_type: LayerType::Dense {
                    input_size: self.layers[i].input_size(),
                    output_size: self.layers[i].output_size(),
                },
                activation: self.activations[i].get_activation_type(),
            };
            layers.push(layer_shape);
        }
        self.shape = NeuralNetworkShape { layers };
    }

    /// gets a subnetwork from the neural network according to the passed shape
    pub fn get_subnetwork(&self, shape: NeuralNetworkShape) -> Option<NeuralNetwork> {
        let mut subnetwork = NeuralNetwork::default();
        let (start, end) = self.deduce_start_end(&shape);
        if start == -1 || end == -1 {
            return None;
        }
        for i in start as usize..end as usize {
            subnetwork
                .add_activation_and_layer(self.activations[i].clone(), self.layers[i].clone());
        }
        subnetwork.deduce_shape();
        Some(subnetwork)
    }

    /// deduce start and end of the subnetwork shape from the neural network
    fn deduce_start_end(&self, shape: &NeuralNetworkShape) -> (i32, i32) {
        for (i, layer) in self.layers.iter().enumerate() {
            // if the layer is not equal to the first one of shape continue
            if layer.input_size() != shape.layers[0].input_size()
                || layer.output_size() != shape.layers[0].output_size()
            {
                continue;
            }
            // if the layer is equal to the first one of shape, check if the rest of the layers are equal
            let mut equal = true;
            for j in 1..shape.layers.len() {
                if i + j >= self.layers.len()
                    || self.layers[i + j].input_size() != shape.layers[j].input_size()
                    || self.layers[i + j].output_size() != shape.layers[j].output_size()
                {
                    equal = false;
                    break;
                }
            }
            if equal {
                let mut to = i + shape.layers.len();
                if to >= self.layers.len() {
                    to = self.layers.len();
                }
                return (i as i32, to as i32);
            }
        }
        (-1, -1)
    }

    /// Adds activation and layer at a specific position in the network.
    fn add_activation_and_layer_at_position(
        &mut self,
        position: usize,
        activation: Box<dyn ActivationTrait + Send>,
        layer: Box<dyn Layer + Send>,
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
