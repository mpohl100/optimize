use super::training_params::TrainingParams;
use super::data_importer::{self, DataImporter};
use crate::neural::nn::neuralnet::NeuralNetwork;

use std::error::Error;

pub struct TrainingSession {
    params: TrainingParams,
    neural_network: NeuralNetwork,
    data_importer: Box<dyn DataImporter>,
}

impl TrainingSession {
    // Constructor
    fn new(params: TrainingParams, data_importer: Box<dyn DataImporter>) -> Result<Self, Box<dyn Error>> {
        // Validate parameters
        if params.num_training_samples() == 0 {
            return Err("Number of training samples must be positive".into());
        }
        if params.num_verification_samples() == 0 {
            return Err("Number of verification samples must be positive".into());
        }
        if params.learning_rate() <= 0.0 {
            return Err("Learning rate must be positive".into());
        }
        if params.learning_rate() >= 1.0 {
            return Err("Learning rate must be less than 1".into());
        }
        if params.epochs() == 0 {
            return Err("Number of epochs must be positive".into());
        }
        if !params.shape().is_valid() {
            return Err("Invalid neural network shape".into());
        }

        let shape = params.shape().clone();
        Ok(Self { params, neural_network: NeuralNetwork::new(shape), data_importer })
    }

    // Train method
    fn train(&mut self) -> Result<f64, Box<dyn Error>> {
        // Prepare the data
        let data = self.data_importer.get_data();
        let inputs = data.data;
        let targets = data.labels;

        // Extract training samples
        if inputs.len() < self.params.num_training_samples() {
            return Err("Not enough training samples".into());
        }

        let training_inputs = inputs[..self.params.num_training_samples()].to_vec();
        let training_targets = targets[..self.params.num_training_samples()].to_vec();

        let input_size = training_inputs[0].len();
        println!("Inputs: {} x {}", inputs.len(), inputs[0].len());
        println!("Targets: {} x {}", targets.len(), targets[0].len());

        // Prepare and validate the neural network
        let nn = &mut self.neural_network;
        if nn.input_size() != input_size {
            return Err("Input size mismatch with neural network".into());
        }
        if nn.output_size() != training_targets[0].len() {
            return Err("Output size mismatch with neural network".into());
        }

        // Train the neural network
        nn.train(&training_inputs, &training_targets, self.params.learning_rate(), self.params.epochs());

        // Verification phase
        let mut success_count = 0;
        for i in 0..self.params.num_verification_samples() {
            let sample_idx = self.params.num_training_samples() + i;
            if sample_idx >= inputs.len() {
                return Err("Not enough verification samples".into());
            }

            let output = nn.predict(inputs[sample_idx].clone());
            let target = &targets[sample_idx];

            // Check if the output matches the target
            if output.iter().zip(target.iter()).all(|(&out, &t)| (out - t).abs() < self.params.tolerance()) {
                success_count += 1;
            }
        }

        // Return the accuracy as the fraction of successful predictions
        Ok(success_count as f64 / self.params.num_verification_samples() as f64)
    }
}