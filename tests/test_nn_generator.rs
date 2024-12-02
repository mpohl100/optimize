use learn::evol::evolution::{EvolutionOptions, LogLevel};
use learn::gen::neuralnet_gen::NeuralNetworkGenerator;
use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::nn::shape::{ActivationType, LayerShape, LayerType};
use learn::neural::training::data_importer::{DataImporter, SessionData};
use learn::neural::training::training_params::TrainingParams;

// Mock DataImporter implementation for testing
#[derive(Clone)]
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
fn test_neural_network_generator() {
    let model_directory = "tests/test_model_generation".to_string();

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

    let training_params = TrainingParams::new(nn_shape.clone(), 0.7, 0.01, 2, 0.1);

    let data_importer = MockDataImporter::new(nn_shape.clone());

    let evolution_params = EvolutionOptions::new(2, LogLevel::Verbose, 3, 4);

    let mut nn_generator = NeuralNetworkGenerator::new(
        training_params,
        evolution_params,
        Box::new(data_importer),
        model_directory.clone(),
    );
    nn_generator.generate();
    nn_generator.save();

    // clear directory
    std::fs::remove_dir_all(model_directory).unwrap();
}
