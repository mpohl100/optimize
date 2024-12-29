use crate::gen::pheno::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::neural::activation::{
    activate::ActivationTrait, relu::ReLU, sigmoid::Sigmoid, softmax::Softmax, tanh::Tanh,
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
            let activation = match layer_shape.activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                ActivationType::Softmax => {
                    Box::new(Softmax::new(layer_shape.activation.temperature().unwrap()))
                        as Box<dyn ActivationTrait + Send>
                }
            };

            network.add_activation_and_layer(activation, layer);
        }

        network
    }

    /// Creates a new `NeuralNetwork` from the given model directory.
    #[allow(clippy::question_mark)]
    pub fn from_disk(model_directory: &String) -> Option<NeuralNetwork> {
        let shape = NeuralNetworkShape::from_disk(model_directory);
        if shape.is_none() {
            return None;
        }
        let sh = shape.unwrap();
        let mut network = NeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: sh.clone(),
        };

        for i in 0..sh.layers.len() {
            let layer = match &sh.layers[i].layer_type() {
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
            let activation = match sh.layers[i].activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU) as Box<dyn ActivationTrait + Send>,
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationTrait + Send>,
                ActivationType::Tanh => Box::new(Tanh) as Box<dyn ActivationTrait + Send>,
                ActivationType::Softmax => {
                    Box::new(Softmax::new(sh.layers[i].activation.temperature().unwrap()))
                        as Box<dyn ActivationTrait + Send>
                }
            };

            network.add_activation_and_layer(activation, layer);
        }

        Some(network)
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
            // this operation should not change the dimension of output
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a forward pass through the network with the given input doing batch caching.
    pub fn forward_batch(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&self.activations) {
            output = layer.forward_batch(&output);
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

    /// Performs a backward pass through the network with the given output gradient doing batch caching.
    pub fn backward_batch(&mut self, grad_output: Vec<f64>) {
        let mut grad = grad_output;
        for (layer, activation) in self
            .layers
            .iter_mut()
            .rev()
            .zip(self.activations.iter_mut().rev())
        {
            grad = activation.backward(&grad);
            grad = layer.backward_batch(&grad);
        }
    }

    /// Trains the neural network using the given inputs, targets, learning rate, and number of epochs.
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
    ) {
        for i in 0..epochs {
            println!("Epoch: {}\r", i);
            let mut loss = 0.0;
            let mut success_count = 0;
            for (input, target) in inputs.iter().zip(targets) {
                // Forward pass
                let output = self.forward(input.as_slice());

                // Check if the output matches the target
                if output
                    .iter()
                    .zip(target.iter())
                    .all(|(&out, &t)| (out - t).abs() < tolerance)
                {
                    success_count += 1;
                }

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
            let accuracy = success_count as f64 / inputs.len() as f64 * 100.0;
            println!("Epoch {}: Loss {}, Accuracy {}%\r", i, loss, accuracy);
        }
    }

    /// Trains the neural network doing batch back propagation.
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    ) {
        for i in 0..epochs {
            println!("Epoch: {}\r", i);
            let mut loss = 0.0;
            let input_chunks = inputs.chunks(batch_size);
            let target_chunks = targets.chunks(batch_size);
            let mut success_count = 0.0;
            for batch in input_chunks.zip(target_chunks) {
                let input_chunk_batch = batch.0;
                let target_chunk_batch = batch.1;
                for (input, target) in input_chunk_batch.iter().zip(target_chunk_batch) {
                    let output = self.forward_batch(input.as_slice());

                    // Check if the output matches the target
                    let mut nb_correct_outputs = 0;
                    for (o, t) in output.iter().zip(target.iter()) {
                        if (o - t).abs() < tolerance {
                            nb_correct_outputs += 1;
                        }
                    }
                    success_count += nb_correct_outputs as f64 / target.len() as f64;

                    let mut grad_output = Vec::new();
                    for j in 0..output.len() {
                        let error = output[j] - target[j];
                        grad_output.push(2.0 * error);
                        loss += error * error;
                    }
                    self.backward_batch(grad_output);
                }
                for layer in &mut self.layers {
                    layer.update_weights(learning_rate);
                }
            }
            let accuracy = success_count / inputs.len() as f64 * 100.0;
            println!(
                "Epoch {}: Loss {}, Accuracy {}%\r",
                i,
                loss / inputs.len() as f64,
                accuracy
            );
            if accuracy < 0.01 && i > 10 {
                break;
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
        let mut nn = NeuralNetwork::new(shape.to_neural_network_shape());
        nn.assign_weights(self);
        *self = nn;
    }

    pub fn assign_weights(&mut self, other: &NeuralNetwork) {
        for i in 0..self.layers.len() {
            if other.layers.len() <= i {
                break;
            }

            self.layers[i].assign_weights(&*other.layers[i]);
        }
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
                activation: self.activations[i].get_activation_data(),
            };
            layers.push(layer_shape);
        }
        self.shape = NeuralNetworkShape { layers };
        if !self.shape.is_valid() {
            println!("Invalid shape: {:?}", self.shape);
            panic!("Invalid shape");
        }
    }

    /// gets a subnetwork from the neural network according to the passed shape
    pub fn get_subnetwork(&self, shape: NeuralNetworkShape) -> Option<NeuralNetwork> {
        if shape.num_layers() == 0 {
            return None;
        }
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
                    activation: ActivationData::new(ActivationType::Sigmoid),
                },
                LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 3,
                        output_size: 3,
                    },
                    activation: ActivationData::new(ActivationType::ReLU),
                },
            ],
        });

        let inputs = vec![vec![1.0, 1.0, 1.0]];
        let targets = vec![vec![0.0, 0.0, 0.0]];

        nn.train(&inputs, &targets, 0.01, 100, 0.1);

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
