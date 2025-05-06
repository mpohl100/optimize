use std::path::Path;

use super::nn_factory::copy_dir_recursive;
use super::nn_factory::neural_network_from_disk;
use super::nn_factory::trainable_neural_network_from_disk;
use super::nn_trait::WrappedNeuralNetwork;
use super::nn_trait::WrappedTrainableNeuralNetwork;
use super::shape::NeuralNetworkShape;

use crate::gen::pheno::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::neural::nn::directory::Directory;
use crate::neural::nn::neuralnet::ClassicNeuralNetwork;
use crate::neural::nn::neuralnet::TrainableClassicNeuralNetwork;
use crate::neural::nn::nn_factory::get_first_free_model_directory;
use crate::neural::nn::nn_trait::NeuralNetwork;
use crate::neural::nn::nn_trait::TrainableNeuralNetwork;
use crate::neural::nn::shape::LayerShape;
use crate::neural::nn::shape::LayerType;

#[derive(Debug)]
pub struct RetryNeuralNetwork {
    primary_nn: WrappedNeuralNetwork,
    backup_nn: WrappedNeuralNetwork,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
}

impl RetryNeuralNetwork {
    pub fn new(shape: NeuralNetworkShape, levels: i32, internal_model_directory: String) -> Self {
        let actual_shape = add_internal_dimensions(shape.clone());
        let primary_nn = WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(
            actual_shape.clone(),
            append_dir(internal_model_directory.clone(), "primary"),
        )));
        let backup_nn = match levels {
            1..=i32::MAX => WrappedNeuralNetwork::new(Box::new(RetryNeuralNetwork::new(
                shape.clone(),
                levels - 1,
                append_dir(internal_model_directory.clone(), "backup"),
            ))),
            0 => WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(
                shape.clone(),
                append_dir(internal_model_directory.clone(), "backup"),
            ))),
            _ => panic!("Invalid level: {}", levels),
        };
        Self {
            primary_nn,
            backup_nn,
            shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_model_directories: vec![],
        }
    }

    pub fn from_disk(model_directory: String) -> WrappedNeuralNetwork {
        let primary_model_directory = append_dir(model_directory.clone(), "primary");
        let backup_model_directory = append_dir(model_directory.clone(), "backup");
        if std::path::Path::new(&primary_model_directory).exists() {
            let primary_nn = WrappedNeuralNetwork::new(Box::new(
                ClassicNeuralNetwork::from_disk(primary_model_directory).unwrap(),
            ));
            let backup_nn = RetryNeuralNetwork::from_disk(backup_model_directory);
            let shape = backup_nn.shape();
            WrappedNeuralNetwork::new(Box::new(Self {
                primary_nn,
                backup_nn,
                shape,
                model_directory: Directory::User(model_directory),
                past_internal_model_directories: vec![],
            }))
        } else {
            WrappedNeuralNetwork::new(Box::new(
                ClassicNeuralNetwork::from_disk(model_directory).unwrap(),
            ))
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let primary_output = self.primary_nn.predict(input.clone());
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if primary_output[primary_output.len() - 1].abs() < 0.05 {
            self.backup_nn.predict(input)
        } else {
            // return the primary output despite the last internal value
            primary_output[0..primary_output.len() - 1].to_vec()
        }
    }
}

fn add_internal_dimensions(shape: NeuralNetworkShape) -> NeuralNetworkShape {
    // Add internal dimensions to the shape
    let mut annotated_shape = AnnotatedNeuralNetworkShape::new(shape.clone());
    let first_layer = shape.layers.first().unwrap();

    // Add internal dimensions to the first layer
    let internal_layer = first_layer.clone();
    let new_dense_layer_type = LayerShape {
        layer_type: LayerType::Dense {
            input_size: internal_layer.input_size(),
            output_size: internal_layer.output_size() + 1,
        },
        activation: internal_layer.activation.clone(),
    };
    annotated_shape.change_layer(0, new_dense_layer_type);

    // Add internal dimensions to the rest of the layers
    for (i, layer) in shape.layers.iter().skip(1).enumerate() {
        // Add internal dimensions to the layer
        let internal_layer = layer.clone();
        let new_dense_layer_type = LayerShape {
            layer_type: LayerType::Dense {
                input_size: internal_layer.input_size() + 1,
                output_size: internal_layer.output_size() + 1,
            },
            activation: internal_layer.activation.clone(),
        };
        annotated_shape.change_layer(i, new_dense_layer_type);
    }
    annotated_shape.to_neural_network_shape()
}

fn append_dir(model_directory: String, subdir: &str) -> String {
    let mut path = model_directory.clone();
    path.push('/');
    path.push_str(subdir);
    path
}

impl NeuralNetwork for RetryNeuralNetwork {
    fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input)
    }

    fn shape(&self) -> NeuralNetworkShape {
        self.shape.clone()
    }

    fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        if let Directory::Internal(_) = self.model_directory {
            self.past_internal_model_directories
                .push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory.clone());
        let primary_user_model_directory = append_dir(user_model_directory.clone(), "primary");
        self.primary_nn.save(primary_user_model_directory)?;
        let backup_user_model_directory = append_dir(user_model_directory, "backup");
        self.backup_nn.save(backup_user_model_directory)?;
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.primary_nn.allocate();
        self.backup_nn.allocate();
    }

    fn deallocate(&mut self) {
        self.primary_nn.deallocate();
        self.backup_nn.deallocate();
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.primary_nn.set_internal();
        self.backup_nn.set_internal();
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        let new_model_directory = get_first_free_model_directory(self.model_directory.clone());
        copy_dir_recursive(
            Path::new(&self.model_directory.path()),
            Path::new(&new_model_directory.clone()),
        )
        .expect("Failed to copy model directory for retry neural network");
        let mut cloned_retry_nn = neural_network_from_disk(new_model_directory);
        cloned_retry_nn.set_internal();
        cloned_retry_nn
    }
}

impl Drop for RetryNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if std::fs::metadata(self.model_directory.path()).is_err() {
            std::fs::create_dir_all(self.model_directory.path()).unwrap();
        }
        self.deallocate();
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        // Remove all past internal model directories
        for dir in &self.past_internal_model_directories {
            if dir != &self.model_directory.path() && std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainableRetryNeuralNetwork {
    primary_nn: WrappedTrainableNeuralNetwork,
    backup_nn: WrappedTrainableNeuralNetwork,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
}

impl TrainableRetryNeuralNetwork {
    pub fn new(shape: NeuralNetworkShape, levels: i32, internal_model_directory: String) -> Self {
        let actual_shape = add_internal_dimensions(shape.clone());
        let primary_nn =
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
                actual_shape.clone(),
                Directory::Internal(append_dir(internal_model_directory.clone(), "primary")),
            )));
        let backup_nn = match levels {
            1..=i32::MAX => {
                WrappedTrainableNeuralNetwork::new(Box::new(TrainableRetryNeuralNetwork::new(
                    shape.clone(),
                    levels - 1,
                    append_dir(internal_model_directory.clone(), "backup"),
                )))
            }
            0 => WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
                shape.clone(),
                Directory::Internal(append_dir(internal_model_directory.clone(), "backup")),
            ))),
            _ => panic!("Invalid level: {}", levels),
        };
        Self {
            primary_nn,
            backup_nn,
            shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_model_directories: vec![],
        }
    }

    pub fn from_disk(model_directory: String) -> WrappedTrainableNeuralNetwork {
        let primary_model_directory = append_dir(model_directory.clone(), "primary");
        let backup_model_directory = append_dir(model_directory.clone(), "backup");
        if std::path::Path::new(&primary_model_directory).exists() {
            let primary_nn = WrappedTrainableNeuralNetwork::new(Box::new(
                TrainableClassicNeuralNetwork::from_disk(primary_model_directory).unwrap(),
            ));
            let backup_nn = TrainableRetryNeuralNetwork::from_disk(backup_model_directory.clone());
            let shape = backup_nn.shape();
            WrappedTrainableNeuralNetwork::new(Box::new(Self {
                primary_nn,
                backup_nn,
                shape,
                model_directory: Directory::User(model_directory),
                past_internal_model_directories: vec![],
            }))
        } else {
            WrappedTrainableNeuralNetwork::new(Box::new(
                TrainableClassicNeuralNetwork::from_disk(model_directory).unwrap(),
            ))
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let primary_output = self.primary_nn.predict(input.clone());
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if primary_output[primary_output.len() - 1].abs() < 0.05 {
            self.backup_nn.predict(input)
        } else {
            // return the primary output despite the last internal value
            primary_output[0..primary_output.len() - 1].to_vec()
        }
    }
}

impl NeuralNetwork for TrainableRetryNeuralNetwork {
    fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input)
    }

    fn shape(&self) -> NeuralNetworkShape {
        self.shape.clone()
    }

    fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        if let Directory::Internal(_) = self.model_directory {
            self.past_internal_model_directories
                .push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory.clone());
        let primary_user_model_directory = append_dir(user_model_directory.clone(), "primary");
        self.primary_nn.save(primary_user_model_directory)?;
        let backup_user_model_directory = append_dir(user_model_directory, "backup");
        self.backup_nn.save(backup_user_model_directory)?;
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.primary_nn.allocate();
        self.backup_nn.allocate();
    }

    fn deallocate(&mut self) {
        self.primary_nn.deallocate();
        self.backup_nn.deallocate();
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.primary_nn.set_internal();
        self.backup_nn.set_internal();
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        unimplemented!()
    }
}

impl TrainableNeuralNetwork for TrainableRetryNeuralNetwork {
    fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
    ) -> f64 {
        // in case one does not have enough samples, don't train and return zero accuracy
        if inputs.len() < 100 {
            return 0.0;
        }
        let mut temp_neural_network = TrainableClassicNeuralNetwork::new(
            self.shape.clone(),
            Directory::Internal(append_dir(self.model_directory.path(), "temp_primary")),
        );
        let _ = temp_neural_network.train(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );

        // evaluate all the samples with the temp neural network
        let mut temp_predictions = Vec::new();
        for input in inputs {
            let prediction = temp_neural_network.predict(input.clone());
            temp_predictions.push(prediction);
        }

        // modify the outputs to contain the information whether they matched or not
        for (output, target) in temp_predictions.iter_mut().zip(targets.iter()) {
            // if the temp_prediction is true, then append a zero to the end of the target
            // Check if the output matches the target
            let mut nb_correct_outputs = 0;
            for (o, t) in output.iter().zip(target.iter()) {
                if (o - t).abs() < tolerance {
                    nb_correct_outputs += 1;
                }
            }
            if nb_correct_outputs == target.len() {
                output.push(0.0);
            } else {
                output.push(1.0);
            }
        }

        // train the primary neural network with the modified outputs
        let primary_accuracy = self.primary_nn.train(
            &temp_predictions,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );

        let mut checked_predictions = Vec::new();
        for input in inputs {
            let prediction = self.primary_nn.predict(input.clone());
            checked_predictions.push(prediction);
        }
        let mut filtered_inputs = Vec::new();
        let mut filtered_targets = Vec::new();
        for (input, prediction) in inputs.iter().zip(checked_predictions.iter()) {
            // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
            if prediction[prediction.len() - 1].abs() > tolerance {
                filtered_inputs.push(input.clone());
                filtered_targets.push(prediction.clone());
            }
        }
        let backup_accuracy = self.backup_nn.train(
            &filtered_inputs,
            &filtered_targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );

        primary_accuracy + backup_accuracy
    }

    fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    ) {
        self.primary_nn.train_batch(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            batch_size,
        );
    }

    fn input_size(&self) -> usize {
        self.shape.layers[0].input_size()
    }

    fn output_size(&self) -> usize {
        self.shape.layers[self.shape.layers.len() - 1].output_size()
    }

    fn duplicate_trainable(&self) -> WrappedTrainableNeuralNetwork {
        let new_model_directory = get_first_free_model_directory(self.model_directory.clone());
        copy_dir_recursive(
            Path::new(&self.model_directory.path()),
            Path::new(&new_model_directory),
        )
        .expect("Failed to copy model directory for trainable retry neural network");
        let mut cloned_retry_nn = trainable_neural_network_from_disk(new_model_directory);
        cloned_retry_nn.set_internal();
        cloned_retry_nn
    }
}

impl Drop for TrainableRetryNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if std::fs::metadata(self.model_directory.path()).is_err() {
            std::fs::create_dir_all(self.model_directory.path()).unwrap();
        }
        self.deallocate();
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.model_directory {
            if std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
        // Remove all past internal model directories
        for dir in &self.past_internal_model_directories {
            if dir != &self.model_directory.path() && std::fs::metadata(dir).is_ok() {
                std::fs::remove_dir_all(dir).unwrap();
            }
        }
    }
}
