use matrix::directory::Directory;
use neural::layer::dense_layer::MatrixParams;
use neural::nn::neuralnet::ClassicNeuralNetwork;
use neural::nn::nn_trait::NeuralNetwork;
use neural::nn::shape::NeuralNetworkShape;
use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;
use neural::utilities::util::{Utils, WrappedUtils};

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

// Helper to generate unique workspace names for tests
static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn create_test_utils() -> WrappedUtils {
    let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let workspace = format!("/tmp/test_workspace_{counter}");
    WrappedUtils::new(Utils::new_with_test_mode(1000000000, 4, workspace))
}

fn cleanup_workspace(utils: &WrappedUtils) {
    let workspace = utils.get_workspace();
    if !workspace.is_empty() && Path::new(&workspace).exists() {
        let _ = std::fs::remove_dir_all(&workspace);
    }
}

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

fn train_model(
    model_directory: String,
    internal_model_directory: String,
    utils: WrappedUtils,
) -> TrainingSession {
    println!("train_model called with model_directory: {model_directory}");
    println!("train_model utils test_mode: {}", utils.is_test_mode());
    println!("train_model utils workspace: {}", utils.get_workspace());

    // Define the neural network shape
    let nn_shape = NeuralNetworkShape {
        layers: vec![
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 30,
                    output_size: 30,
                    matrix_params: MatrixParams { slice_rows: 50, slice_cols: 50 },
                },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
            LayerShape {
                layer_type: LayerType::Stretch {
                    input_size: 30,
                    output_size: 2,
                    matrix_params: MatrixParams { slice_rows: 50, slice_cols: 50 },
                },
                activation: ActivationData::new(ActivationType::ReLU),
            },
        ],
    };

    // Define training parameters
    let training_params =
        TrainingParams::new(nn_shape.clone(), None, None, 0.7, 0.01, 3, 0.1, 32, true, 1.0);

    // Create a training session using the mock data importer
    let data_importer = MockDataImporter::new(nn_shape);

    let training_session = TrainingSession::from_disk(
        model_directory.clone(),
        training_params.clone(),
        Box::new(data_importer.clone()),
        utils.clone(),
    );
    match training_session {
        Ok(mut training_session) => {
            // Train the neural network and check the success rate
            let success_rate = training_session.train().expect("Training failed");
            // print the success rate
            println!("Success rate: {success_rate}");
            training_session.save_model(model_directory.clone()).expect("Failed to save model");
            return training_session;
        },
        Err(e) => {
            println!("Failed to load model: {e}");
        },
    }

    let mut training_session = TrainingSession::new(
        training_params,
        Box::new(data_importer),
        &Directory::Internal(internal_model_directory),
        utils.clone(),
    )
    .expect("Failed to create TrainingSession");

    // Train the neural network and check the success rate
    let success_rate = training_session.train().expect("Training failed");
    // print the success rate
    println!("Success rate: {success_rate}");
    training_session
}

#[test]
fn test_training_session() {
    let utils = create_test_utils();
    let model_directory = format!("{}/model_dir", utils.get_workspace());
    let internal_model_directory = format!("{}/internal_model_dir", utils.get_workspace());
    let training_session =
        train_model(model_directory.clone(), internal_model_directory.clone(), utils.clone());
    // Clean up workspace after test
    cleanup_workspace(&utils);
}
