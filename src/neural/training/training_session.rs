use super::data_importer::DataImporter;
use super::training_params::TrainingParams;
use crate::neural::nn::neuralnet::NeuralNetwork;

use std::error::Error;

pub struct TrainingSession {
    params: TrainingParams,
    neural_network: NeuralNetwork,
    data_importer: Box<dyn DataImporter>,
}

impl TrainingSession {
    // Constructor
    pub fn new(
        params: TrainingParams,
        data_importer: Box<dyn DataImporter>,
    ) -> Result<Self, Box<dyn Error>> {
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
        Ok(Self {
            params,
            neural_network: NeuralNetwork::new(shape),
            data_importer,
        })
    }

    // Train method
    pub fn train(&mut self) -> Result<f64, Box<dyn Error>> {
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
        nn.train(
            &training_inputs,
            &training_targets,
            self.params.learning_rate(),
            self.params.epochs(),
        );

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
            if output
                .iter()
                .zip(target.iter())
                .all(|(&out, &t)| (out - t).abs() < self.params.tolerance())
            {
                success_count += 1;
            }
        }

        // Return the accuracy as the fraction of successful predictions
        Ok(success_count as f64 / self.params.num_verification_samples() as f64)
    }

    pub fn save_model(&self, model_directory: String) -> Result<(), Box<dyn Error>> {
        let shape = self.neural_network.shape();
        shape.to_yaml(model_directory.clone());
        self.neural_network.save_layers(model_directory)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::nn::shape::NeuralNetworkShape;
    use crate::neural::nn::shape::{ActivationType, LayerShape, LayerType};
    use crate::neural::training::data_importer::{DataImporter, SessionData};

    // Mock DataImporter implementation for testing
    struct MockDataImporter {
        shape: NeuralNetworkShape,
    }

    impl MockDataImporter {
        fn new(shape: NeuralNetworkShape) -> Self {
            Self { shape }
        }
    }

    impl DataImporter for MockDataImporter {
        fn get_data(&self) -> SessionData {
            let num_samples = 1000;
            let input_size = self.shape.layers[0].input_size(); // Input dimension (e.g., 28x28 image flattened)
            let num_classes = self.shape.layers[self.shape.layers.len() - 1].output_size(); // Number of output classes (e.g., for digit classification)

            // Initialize inputs and targets with zeros
            let data = vec![vec![0.0; input_size]; num_samples];
            let labels = vec![vec![0.0; num_classes]; num_samples];

            SessionData { data, labels }
        }
    }

    #[test]
    fn test_simple_neural_net() {
        // Define the neural network shape
        let nn_shape = NeuralNetworkShape {
            layers: vec![
                LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 128,
                        output_size: 128,
                    },
                    activation: ActivationType::ReLU,
                },
                LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 128,
                        output_size: 64,
                    },
                    activation: ActivationType::ReLU,
                },
                LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 64,
                        output_size: 10,
                    },
                    activation: ActivationType::Sigmoid,
                },
            ],
        };

        // Define training parameters
        let training_params = TrainingParams::new(nn_shape.clone(), 700, 300, 0.01, 10, 0.1);

        // Create a training session using the mock data importer
        let data_importer = MockDataImporter::new(nn_shape);

        let mut training_session = TrainingSession::new(training_params, Box::new(data_importer))
            .expect("Failed to create TrainingSession");

        // Train the neural network and check the success rate
        let success_rate = training_session.train().expect("Training failed");
        assert!(
            success_rate >= 0.9,
            "Expected success rate >= 0.9, got {}",
            success_rate
        );
    }
}
