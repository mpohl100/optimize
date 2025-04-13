use crate::gen::pheno::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::neural::activation::{
    activate::ActivationTrait, relu::ReLU, sigmoid::Sigmoid, softmax::Softmax, tanh::Tanh,
};
use crate::neural::layer::dense_layer::DenseLayer;
use crate::neural::layer::dense_layer::TrainableDenseLayer;
use crate::neural::layer::layer_trait::WrappedLayer;
use crate::neural::layer::layer_trait::WrappedTrainableLayer;
use crate::neural::nn::shape::*;

use indicatif::ProgressDrawTarget;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use once_cell::sync::Lazy;
use std::sync::Arc;
use std::path::Path;
use std::fs;
use std::io;

// Create a static MultiProgress instance
static MULTI_PROGRESS: Lazy<Arc<MultiProgress>> = Lazy::new(|| Arc::new(MultiProgress::new()));

use std::boxed::Box;

use super::directory::Directory;

/// A neural network.
#[derive(Debug, Default)]
pub struct NeuralNetwork {
    layers: Vec<WrappedLayer>,
    activations: Vec<Box<dyn ActivationTrait + Send>>,
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_directory: Vec<String>,
}

impl NeuralNetwork {
    /// Creates a new `NeuralNetwork` from the given shape.
    pub fn new(shape: NeuralNetworkShape, internal_model_directory: String) -> Self {
        let shape_clone = shape.clone();
        let mut network = NeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_directory: Vec::new(),
        };

        // Initialize layers and activations based on the provided shape.
        for (i, layer_shape) in shape_clone.layers.iter().enumerate() {
            // Here you would instantiate the appropriate Layer and Activation objects.
            let dense_layer = DenseLayer::new(layer_shape.input_size(), layer_shape.output_size(), network.model_directory.clone(), i);
            let layer = WrappedLayer::new(Box::new(dense_layer));
            let activation = match layer_shape.activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
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

    /// Makes a prediction based on a single input by performing a forward pass.
    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input.as_slice())
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
            model_directory: Directory::User(model_directory.clone()),
            past_internal_directory: Vec::new(),
        };

        for i in 0..sh.layers.len() {
            let layer = match &sh.layers[i].layer_type() {
                LayerType::Dense {
                    input_size,
                    output_size,
                } => {
                    let layer = DenseLayer::new(*input_size, *output_size, network.model_directory.clone(), i);
                    WrappedLayer::new(Box::new(layer))
                }
            };
            let activation = match sh.layers[i].activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
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
        layer: WrappedLayer,
    ) {
        self.activations.push(activation);
        self.layers.push(layer);
    }

    /// Performs a forward pass through the network with the given input.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
            layer.allocate();
            layer.mark_for_use();
            output = layer.forward(&output);
            layer.free_from_use();
            // this operation should not change the dimension of output
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a forward pass through the network with the given input doing batch caching.
    pub fn forward_batch(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
            output = layer.forward_batch(&output);
            output = activation.forward(&output);
        }
        output
    }

    pub fn shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    pub fn save_layers(&self, model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        // make a layers subdirectory
        std::fs::create_dir_all(format!("{}/layers", model_directory))?;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.save(format!("{}/layers/layer_{}.txt", model_directory, i))?;
        }
        Ok(())
    }

    pub fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        self.past_internal_directory.push(self.model_directory.path());
        self.model_directory = Directory::User(user_model_directory.clone());
        let model_directory = self.model_directory.path();
        self.save_internal(model_directory.clone())
    }

    fn save_internal(&self, model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        // remove the directory if it exists
        let backup_directory = format!("{}_backup", model_directory);
        if std::fs::metadata(&model_directory).is_ok() {
            // copy the directory to a backup
            copy_dir_recursive(Path::new(&model_directory), Path::new(&backup_directory))?;
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

    pub fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    pub fn deallocate(&mut self) {
        for layer in &mut self.layers {
            layer.deallocate();
        }
    }

    pub fn save_layout(&self){
        let shape = self.shape();
        shape.to_yaml(self.model_directory.path());
    }

    fn get_first_free_model_directory(&self) -> String {
        let model_directory_orig = self.model_directory.path();
        // truncate _{integer} from the end of the model_directory
        let mut model_directory = model_directory_orig.clone();
        if let Some(pos) = model_directory.rfind('_') {
            // check that the remainder is an integer
            let remainder = &model_directory[pos + 1..];
            if remainder.parse::<usize>().is_ok() {
                model_directory = model_directory[..pos].to_string();
            }
        }

        let mut i = 1;
        while std::fs::metadata(format!("{}_{}", model_directory, i)).is_ok() {
            i += 1;
        }
        model_directory = format!("{}_{}", model_directory, i);
        // create the directory to block the name
        std::fs::create_dir_all(&model_directory).unwrap();
        model_directory
    }
}

impl Clone for NeuralNetwork {
    fn clone(&self) -> Self {
        // create a sibling directory with the postfix _clone appendended to model_direcotory path
        let model_directory = self.get_first_free_model_directory();
        // Save the model to the new directory
        self.save_internal(model_directory.clone()).unwrap();
        // Clone the neural network by cloning its layers and activations
        let mut new_layers = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            new_layers.push(layer.duplicate(model_directory.clone(), i));
            layer.cleanup();
        }
        NeuralNetwork {
            layers: new_layers,
            activations: self.activations.clone(),
            shape: self.shape.clone(),
            model_directory: Directory::Internal(model_directory),
            past_internal_directory: Vec::new(),
        }
    }
}

impl Drop for NeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if !std::fs::metadata(self.model_directory.path()).is_ok() {
            std::fs::create_dir_all(self.model_directory.path()).unwrap();
        }
        self.save_layout();
        self.deallocate();
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        // Remove all past internal model directories
        for dir in &self.past_internal_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}

/// A neural network.
#[derive(Debug, Default)]
pub struct TrainableNeuralNetwork {
    layers: Vec<WrappedTrainableLayer>,
    activations: Vec<Box<dyn ActivationTrait + Send>>,
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directory: Vec<String>,
}

impl TrainableNeuralNetwork {
    /// Creates a new `NeuralNetwork` from the given shape.
    pub fn new(shape: NeuralNetworkShape, model_directory: Directory) -> Self {
        let shape_clone = shape.clone();
        let mut network = TrainableNeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape,
            model_directory: model_directory,
            past_internal_model_directory: Vec::new(),
        };

        // Initialize layers and activations based on the provided shape.
        for (i, layer_shape) in shape_clone.layers.iter().enumerate() {
            // Here you would instantiate the appropriate Layer and Activation objects.
            let layer = WrappedTrainableLayer::new(Box::new(TrainableDenseLayer::new(
                layer_shape.input_size(),
                layer_shape.output_size(),
                network.model_directory.clone(),
                i,
            )));
            let activation = match layer_shape.activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
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

    pub fn new_dir(directory: Directory) -> Self {
        TrainableNeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: NeuralNetworkShape::default(),
            model_directory: directory,
            past_internal_model_directory: Vec::new(),
        }
    }

    /// Creates a new `NeuralNetwork` from the given model directory.
    #[allow(clippy::question_mark)]
    pub fn from_disk(model_directory: &String) -> Option<TrainableNeuralNetwork> {
        let shape = NeuralNetworkShape::from_disk(model_directory);
        if shape.is_none() {
            return None;
        }
        let sh = shape.unwrap();
        let mut network = TrainableNeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape: sh.clone(),
            model_directory: Directory::User(model_directory.clone()),
            past_internal_model_directory: Vec::new(),
        };

        for i in 0..sh.layers.len() {
            let layer = match &sh.layers[i].layer_type() {
                LayerType::Dense {
                    input_size,
                    output_size,
                } => {
                    let layer = TrainableDenseLayer::new(*input_size, *output_size, network.model_directory.clone(), i);
                    WrappedTrainableLayer::new(Box::new(layer))
                }
            };
            let activation = match sh.layers[i].activation.activation_type() {
                ActivationType::ReLU => Box::new(ReLU::new()) as Box<dyn ActivationTrait + Send>,
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
        layer: WrappedTrainableLayer,
    ) {
        self.activations.push(activation);
        self.layers.push(layer);
    }

    /// Performs a forward pass through the network with the given input.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
            layer.allocate();
            layer.mark_for_use();
            output = layer.forward(&output);
            layer.free_from_use();
            // this operation should not change the dimension of output
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a forward pass through the network with the given input doing batch caching.
    pub fn forward_batch(&mut self, input: &[f64]) -> Vec<f64> {
        let mut output = input.to_vec();
        for (layer, activation) in self.layers.iter_mut().zip(&mut self.activations) {
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
    /// Includes validation using a split of the data.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
    ) {
        assert!(
            (0.0..=1.0).contains(&validation_split),
            "validation_split must be between 0 and 1"
        );

        let split_index = (inputs.len() as f64 * validation_split).round() as usize;
        let (train_inputs, validation_inputs) = inputs.split_at(split_index);
        let (train_targets, validation_targets) = targets.split_at(split_index);

        let multi_progress = Arc::clone(&MULTI_PROGRESS);

        for epoch in 0..epochs {
            // Initialize progress bar
            let pb = multi_progress.add(ProgressBar::new(train_inputs.len() as u64));
            pb.set_draw_target(ProgressDrawTarget::stdout());
            pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} | {msg}")
                .expect("Invalid template")
                .progress_chars("#>-"),);

            let mut loss = 0.0;
            let mut success_count = 0.0;

            train_inputs
                .iter()
                .zip(train_targets)
                .enumerate()
                .for_each(|(j, (input, target))| {
                    // Forward pass
                    let output = self.forward(input.as_slice());

                    // Calculate accuracy
                    let correct_outputs = output
                        .iter()
                        .zip(target.iter())
                        .filter(|(&o, &t)| (o - t).abs() < tolerance)
                        .count();
                    success_count += correct_outputs as f64 / target.len() as f64;

                    // Calculate loss gradient
                    let grad_output: Vec<f64> = output
                        .iter()
                        .zip(target)
                        .map(|(o, t)| {
                            let error = o - t;
                            loss += error * error;
                            2.0 * error
                        })
                        .collect();

                    // Backward pass
                    self.backward(grad_output);

                    // Update weights
                    if use_adam {
                        self.adjust_adam(j + 1, learning_rate, 0.9, 0.999, 1e-8);
                    } else {
                        self.layers
                            .iter_mut()
                            .for_each(|layer| layer.update_weights(learning_rate));
                    }

                    // Update the progress bar
                    let accuracy = success_count / train_inputs.len() as f64 * 100.0;
                    let loss_display = loss / train_inputs.len() as f64;
                    pb.set_position((j + 1) as u64);
                    pb.set_message(format!(
                        "Accuracy: {:.2} %, Loss: {:.4}",
                        accuracy, loss_display
                    ));
                });

            // Validation phase
            let mut validation_loss = 0.0;
            let mut validation_success_count = 0.0;

            validation_inputs
                .iter()
                .zip(validation_targets)
                .for_each(|(input, target)| {
                    let output = self.forward(input.as_slice());
                    let correct_outputs = output
                        .iter()
                        .zip(target.iter())
                        .filter(|(&o, &t)| (o - t).abs() < tolerance)
                        .count();
                    validation_success_count += correct_outputs as f64 / target.len() as f64;

                    validation_loss += output
                        .iter()
                        .zip(target)
                        .map(|(o, t)| {
                            let error = o - t;
                            error * error
                        })
                        .sum::<f64>();
                });

            validation_loss /= validation_inputs.len() as f64;
            let validation_accuracy =
                validation_success_count / validation_inputs.len() as f64 * 100.0;

            // Finish the progress bar
            loss /= train_inputs.len() as f64;
            let accuracy = success_count / train_inputs.len() as f64 * 100.0;
            let message = format!(
            "Epoch {} finished | Train Acc: {:.2} %, Train Loss: {:.4} | Val Acc: {:.2} %, Val Loss: {:.4}",
            epoch, accuracy, loss, validation_accuracy, validation_loss);
            pb.finish_with_message(message);
            multi_progress.remove(&pb);
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
            layer.save_weight(format!("{}/layers/layer_{}.txt", model_directory, i))?;
        }
        Ok(())
    }

    pub fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        self.past_internal_model_directory.push(self.model_directory.path());
        self.model_directory = Directory::User(user_model_directory.clone());
        let model_directory = self.model_directory.path();
        self.save_internal(model_directory.clone())
    }

    fn save_internal(&self, model_directory: String) -> Result<(), Box<dyn std::error::Error>> {

        // remove the directory if it exists
        let backup_directory = format!("{}_backup", model_directory);
        if std::fs::metadata(&model_directory).is_ok() {
            // copy the directory to a backup not move
            copy_dir_recursive(Path::new(&model_directory), Path::new(&backup_directory))?;
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
        let mut nn = TrainableNeuralNetwork::new(shape.to_neural_network_shape(), self.model_directory.clone());
        nn.assign_weights(self);
        *self = nn;
    }

    pub fn assign_weights(&mut self, other: &TrainableNeuralNetwork) {
        for i in 0..self.layers.len() {
            if other.layers.len() <= i {
                break;
            }

            self.layers[i].assign_weights(other.layers[i].clone());
        }
    }

    pub fn merge(&self, other: TrainableNeuralNetwork) -> TrainableNeuralNetwork {
        // Rethink this function entirely
        let mut new_nn = TrainableNeuralNetwork::new_dir(Directory::Internal(self.get_first_free_model_directory()));
        for i in 0..self.layers.len() {
            new_nn.add_activation_and_layer(self.activations[i].clone(), self.layers[i].clone());
        }

        let merge_layer_input_size = self.layers.last().unwrap().output_size();
        let merge_layer_output_size = other.layers.first().unwrap().input_size();
        let merge_layer = WrappedTrainableLayer::new(Box::new(TrainableDenseLayer::new(
            merge_layer_input_size,
            merge_layer_output_size,
            new_nn.model_directory.clone(),
            new_nn.layers.len(),
        )));
        let merge_activation = Box::new(ReLU::new());
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
    pub fn get_subnetwork(&self, shape: NeuralNetworkShape) -> Option<TrainableNeuralNetwork> {
        if shape.num_layers() == 0 {
            return None;
        }
        let mut subnetwork = TrainableNeuralNetwork::default();
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

    fn adjust_adam(&mut self, t: usize, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) {
        for layer in &mut self.layers {
            layer.adjust_adam(t, learning_rate, beta1, beta2, epsilon);
        }
    }

    pub fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    pub fn deallocate(&mut self) {
        for layer in &mut self.layers {
            layer.deallocate();
        }
    }

    pub fn allocate(&mut self) {
        for layer in &mut self.layers {
            layer.allocate();
        }
    }

    pub fn save_layout(&self){
        let shape = self.shape();
        shape.to_yaml(self.model_directory.path());
    }

    fn get_first_free_model_directory(&self) -> String {
        let model_directory_orig = self.model_directory.path();
        // truncate _{integer} from the end of the model_directory
        let mut model_directory = model_directory_orig.clone();
        if let Some(pos) = model_directory.rfind('_') {
            // check that the remainder is an integer
            let remainder = &model_directory[pos + 1..];
            if remainder.parse::<usize>().is_ok() {
                model_directory = model_directory[..pos].to_string();
            }
        }
        let mut i = 1;
        while std::fs::metadata(format!("{}_{}", model_directory, i)).is_ok() {
            i += 1;
        }
        model_directory = format!("{}_{}", model_directory, i);
        // create the directory to block the name
        std::fs::create_dir_all(&model_directory).unwrap();
        model_directory
    }
}

impl Clone for TrainableNeuralNetwork {
    fn clone(&self) -> Self {
        // create a sibling directory with the postfix _clone appendended to model_direcotory path
        let model_directory = self.get_first_free_model_directory();
        // Save the model to the new directory
        self.save_internal(model_directory.clone()).unwrap();
        // Clone the neural network by cloning its layers and activations
        let mut new_layers = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            new_layers.push(layer.duplicate(model_directory.clone(), i));
            layer.cleanup();
        }
        TrainableNeuralNetwork {
            layers: new_layers,
            activations: self.activations.clone(),
            shape: self.shape.clone(),
            model_directory: Directory::Internal(model_directory),
            past_internal_model_directory: Vec::new(),
        }
    }
}

impl Drop for TrainableNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if !std::fs::metadata(self.model_directory.path()).is_ok() {
            std::fs::create_dir_all(self.model_directory.path()).unwrap();
        }
        self.save_layout();
        self.deallocate();
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        // Remove all past internal model directories
        for dir in &self.past_internal_model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry_result in fs::read_dir(src)? {
        let entry = entry_result?;
        let path = entry.path();
        let dest_path = dst.join(entry.file_name());

        if path.is_dir() {
            copy_dir_recursive(&path, &dest_path)?;
        } else {
            fs::copy(&path, &dest_path)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::nn::shape::{ActivationType, LayerShape};

    #[test]
    fn test_neural_network_train() {
        let mut nn = TrainableNeuralNetwork::new(NeuralNetworkShape {
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
        }, 
        Directory::Internal("internal_model".to_string()));

        let inputs = vec![vec![1.0, 1.0, 1.0]];
        let targets = vec![vec![0.0, 0.0, 0.0]];

        nn.train(&inputs, &targets, 0.01, 100, 0.1, true, 0.7);

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
