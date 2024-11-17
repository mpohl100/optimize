use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::nn::shape::{ActivationType, LayerShape, LayerType};
use learn::neural::training::data_importer::{DataImporter, SessionData};
use learn::neural::training::training_params::TrainingParams;
use learn::neural::training::training_session::TrainingSession;

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

fn main() {
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
    // print the success rate
    println!("Success rate: {}", success_rate);
    assert!(
        success_rate >= 0.9,
        "Expected success rate >= 0.9, got {}",
        success_rate
    );
}
