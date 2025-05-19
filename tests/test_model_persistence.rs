use learn::neural::nn::directory::Directory;
use learn::neural::nn::neuralnet::ClassicNeuralNetwork;
use learn::neural::nn::nn_trait::NeuralNetwork;
use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
use learn::neural::training::data_importer::{DataImporter, SessionData};
use learn::neural::training::training_params::TrainingParams;
use learn::neural::training::training_session::TrainingSession;
use learn::neural::utilities::util::{Utils, WrappedUtils};

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

fn train_model(model_directory: String, internal_model_directory: String) {
    // Define the neural network shape
    let nn_shape = NeuralNetworkShape {
        layers: vec![
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 128,
                    output_size: 128,
                },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 128,
                    output_size: 64,
                },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 64,
                    output_size: 64,
                },
                activation: ActivationData::new_softmax(2.0),
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 64,
                    output_size: 10,
                },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
        ],
    };

    // Define training parameters
    let training_params = TrainingParams::new(nn_shape.clone(), None, None, 0.7, 0.01, 10, 0.1, 32, true);

    // Create a training session using the mock data importer
    let data_importer = MockDataImporter::new(nn_shape);

    let utils = WrappedUtils::new(Utils::new(1000000000, 4));

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
            println!("Success rate: {}", success_rate);
            training_session
                .save_model(model_directory.clone())
                .expect("Failed to save model");
            return;
        }
        Err(e) => {
            println!("Failed to load model: {}", e);
        }
    }

    let mut training_session = TrainingSession::new(
        training_params,
        Box::new(data_importer),
        Directory::Internal(internal_model_directory),
        utils.clone(),
    )
    .expect("Failed to create TrainingSession");

    // Train the neural network and check the success rate
    let success_rate = training_session.train().expect("Training failed");
    // print the success rate
    println!("Success rate: {}", success_rate);
    training_session
        .save_model(model_directory.clone())
        .expect("Failed to save model");
}

#[test]
fn new_model_is_persisted() {
    // Arrange
    let model_directory = "tests/test_model_persistence_1".to_string();

    // Act
    train_model(
        model_directory.clone(),
        "tests/test_model_persistence_1_internal".to_string(),
    );

    // Assert
    // Check if the model directory exists
    assert!(std::path::Path::new(&model_directory).exists());

    // Clean up
    std::fs::remove_dir_all(&model_directory).unwrap();
}

#[test]
fn already_trained_model_is_loaded() {
    // Arrange
    let model_directory = "tests/test_model_persistence_2".to_string();
    train_model(
        model_directory.clone(),
        "tests/test_model_persistence_2_internal".to_string(),
    );

    // Act
    train_model(
        model_directory.clone(),
        "tests/test_model_persistence_3_internal".to_string(),
    );

    // Assert
    // Check if the model directory exists
    assert!(std::path::Path::new(&model_directory).exists());
    // Check that backup directory is removed
    assert!(!std::path::Path::new(&format!("{}_backup", model_directory)).exists());

    // Clean up
    std::fs::remove_dir_all(&model_directory).unwrap();
    // Remove the backup directory if it exists
    if std::path::Path::new(&format!("{}_backup", model_directory)).exists() {
        std::fs::remove_dir_all(&format!("{}_backup", model_directory)).unwrap();
    }
}

#[test]
fn trained_model_is_convertible_to_ordinary_model_and_back() {
    let model_directory = "tests/test_model_persistence_3".to_string();
    let new_model_directory = "tests/test_model_persistence_3_new".to_string();
    {
        // Arrange
        let utils = WrappedUtils::new(Utils::new(1000000000, 4));

        train_model(
            model_directory.clone(),
            "tests/test_model_persistence_3_internal".to_string(),
        );

        let mut ordinary_model =
            ClassicNeuralNetwork::from_disk(model_directory.clone(), utils.clone()).unwrap();
        ordinary_model.allocate();
        // Act
        ordinary_model
            .save(new_model_directory.clone())
            .expect("Failed to save model");

        let mut trainable_ordinary_model =
            ClassicNeuralNetwork::from_disk(new_model_directory.clone(), utils).unwrap();

        trainable_ordinary_model.allocate();

        trainable_ordinary_model
            .save(new_model_directory.clone())
            .expect("Failed to save trainable model");
    }
    // Assert
    // Check if the new model directory exists
    assert!(std::path::Path::new(&new_model_directory).exists());
    // Check that the backup directory is removed
    assert!(!std::path::Path::new(&format!("{}_backup", model_directory)).exists());
    // Check that the backup directory is removed
    assert!(!std::path::Path::new(&format!("{}_backup", new_model_directory)).exists());
    // Clean up
    std::fs::remove_dir_all(&new_model_directory).unwrap();
}
