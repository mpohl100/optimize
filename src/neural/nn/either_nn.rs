use std::path::Path;

use super::nn_factory::copy_dir_recursive;
use super::nn_factory::neural_network_from_disk;
use super::nn_factory::new_trainable_neural_network;
use super::nn_factory::trainable_neural_network_from_disk;
use super::nn_factory::NeuralNetworkCreationArguments;
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
use crate::neural::utilities::util::WrappedUtils;

#[derive(Debug)]
pub struct EitherNeuralNetwork {
    pre_nn: WrappedNeuralNetwork,
    left_nn: Option<WrappedNeuralNetwork>,
    right_nn: Option<WrappedNeuralNetwork>,
    shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
    utils: WrappedUtils,
}

impl EitherNeuralNetwork {
    pub fn from_disk(model_directory: String, utils: WrappedUtils) -> WrappedNeuralNetwork {
        let pre_model_directory = append_dir(model_directory.clone(), "pre");
        let left_model_directory = append_dir(model_directory.clone(), "left");
        let right_model_directory = append_dir(model_directory.clone(), "right");
        if std::path::Path::new(&pre_model_directory).exists() {
            let pre_nn = neural_network_from_disk(pre_model_directory, utils.clone());

            let left_nn = if std::path::Path::new(&left_model_directory).exists() {
                Some(neural_network_from_disk(
                    left_model_directory,
                    utils.clone(),
                ))
            } else {
                None
            };

            let right_nn = if std::path::Path::new(&right_model_directory).exists() {
                Some(neural_network_from_disk(
                    right_model_directory,
                    utils.clone(),
                ))
            } else {
                None
            };

            let shape = pre_nn.shape();
            WrappedNeuralNetwork::new(Box::new(Self {
                pre_nn,
                left_nn,
                right_nn,
                shape,
                model_directory: Directory::User(model_directory),
                past_internal_model_directories: vec![],
                utils,
            }))
        } else {
            WrappedNeuralNetwork::new(Box::new(
                ClassicNeuralNetwork::from_disk(model_directory, utils).unwrap(),
            ))
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let pre_output = self.pre_nn.predict(input.clone());
        if self.left_nn.is_none() && self.right_nn.is_none() {
            return pre_output;
        }
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if (pre_output[0] - 1.0).abs() < 0.2 {
            if self.left_nn.is_none() {
                return pre_output;
            }
            let mut left_nn = self.left_nn.as_ref().unwrap().clone();

            return left_nn.predict(input.clone());
        } else {
            if self.right_nn.is_none() {
                return pre_output;
            }
            let mut right_nn = self.right_nn.as_ref().unwrap().clone();
            return right_nn.predict(input.clone());
        }
    }
}

fn append_dir(model_directory: String, subdir: &str) -> String {
    let mut path = model_directory.clone();
    path.push('/');
    path.push_str(subdir);
    path
}

impl NeuralNetwork for EitherNeuralNetwork {
    fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input)
    }

    fn shape(&self) -> NeuralNetworkShape {
        if self.left_nn.is_some() {
            self.left_nn.as_ref().unwrap().shape()
        } else if self.right_nn.is_some() {
            self.right_nn.as_ref().unwrap().shape()
        } else {
            self.shape.clone()
        }
    }

    fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        if let Directory::Internal(_) = self.model_directory {
            self.past_internal_model_directories
                .push(self.model_directory.path());
        }
        self.model_directory = Directory::User(user_model_directory.clone());
        let pre_user_model_directory = append_dir(user_model_directory.clone(), "pre");
        self.pre_nn.save(pre_user_model_directory)?;

        match self.left_nn {
            Some(ref mut left_nn) => {
                let left_user_model_directory = append_dir(user_model_directory.clone(), "left");
                left_nn.save(left_user_model_directory)?;
            }
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => {
                let right_user_model_directory = append_dir(user_model_directory, "right");
                right_nn.save(right_user_model_directory)?;
            }
            None => {}
        }
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.pre_nn.allocate();
        match self.left_nn {
            Some(ref mut left_nn) => left_nn.allocate(),
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => right_nn.allocate(),
            None => {}
        }
    }

    fn deallocate(&mut self) {
        self.pre_nn.deallocate();
        match self.left_nn {
            Some(ref mut left_nn) => left_nn.deallocate(),
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => right_nn.deallocate(),
            None => {}
        }
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.pre_nn.set_internal();
        match self.left_nn {
            Some(ref mut left_nn) => left_nn.set_internal(),
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => right_nn.set_internal(),
            None => {}
        }
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        let new_model_directory = get_first_free_model_directory(self.model_directory.clone());
        copy_dir_recursive(
            Path::new(&self.model_directory.path()),
            Path::new(&new_model_directory.clone()),
        )
        .expect("Failed to copy model directory for retry neural network");
        let mut cloned_retry_nn = neural_network_from_disk(new_model_directory, self.utils.clone());
        cloned_retry_nn.set_internal();
        cloned_retry_nn
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl Drop for EitherNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if let Directory::User(_) = &self.model_directory {
            if std::fs::metadata(self.model_directory.path()).is_err() {
                std::fs::create_dir_all(self.model_directory.path()).unwrap();
            }
            self.deallocate();
        }
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
pub struct TrainableEitherNeuralNetwork {
    pre_nn: WrappedTrainableNeuralNetwork,
    left_nn: Option<WrappedTrainableNeuralNetwork>,
    right_nn: Option<WrappedTrainableNeuralNetwork>,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    pre_shape: NeuralNetworkShape,
    model_directory: Directory,
    past_internal_model_directories: Vec<String>,
    utils: WrappedUtils,
}

impl TrainableEitherNeuralNetwork {
    pub fn new(
        shape: NeuralNetworkShape,
        pre_shape: NeuralNetworkShape,
        internal_model_directory: String,
        utils: WrappedUtils,
    ) -> Self {
        let pre_nn =
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
                shape.clone(),
                Directory::Internal(append_dir(internal_model_directory.clone(), "pre")),
                utils.clone(),
            )));
        Self {
            pre_nn,
            left_nn: None,
            right_nn: None,
            shape,
            pre_shape,
            model_directory: Directory::Internal(internal_model_directory),
            past_internal_model_directories: vec![],
            utils,
        }
    }

    pub fn from_disk(
        model_directory: String,
        utils: WrappedUtils,
    ) -> WrappedTrainableNeuralNetwork {
        let pre_model_directory = append_dir(model_directory.clone(), "pre");
        let primary_model_directory = append_dir(model_directory.clone(), "left");
        let backup_model_directory = append_dir(model_directory.clone(), "right");
        if std::path::Path::new(&pre_model_directory).exists() {
            let pre_nn = trainable_neural_network_from_disk(pre_model_directory, utils.clone());

            let left_nn = if std::path::Path::new(&primary_model_directory).exists() {
                Some(trainable_neural_network_from_disk(
                    primary_model_directory,
                    utils.clone(),
                ))
            } else {
                None
            };

            let right_nn = if std::path::Path::new(&backup_model_directory).exists() {
                Some(trainable_neural_network_from_disk(
                    backup_model_directory,
                    utils.clone(),
                ))
            } else {
                None
            };

            let shape = pre_nn.shape();
            let pre_shape = pre_nn.shape();
            WrappedTrainableNeuralNetwork::new(Box::new(Self {
                pre_nn,
                left_nn,
                right_nn,
                shape,
                pre_shape,
                model_directory: Directory::User(model_directory),
                past_internal_model_directories: vec![],
                utils,
            }))
        } else {
            WrappedTrainableNeuralNetwork::new(Box::new(
                TrainableClassicNeuralNetwork::from_disk(model_directory, utils).unwrap(),
            ))
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let pre_output = self.pre_nn.predict(input.clone());
        if self.left_nn.is_none() && self.right_nn.is_none() {
            return pre_output;
        }
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if (pre_output[0] - 1.0).abs() < 0.2 {
            if self.left_nn.is_none() {
                return pre_output;
            }
            let mut left_nn = self.left_nn.as_ref().unwrap().clone();

            return left_nn.predict(input.clone());
        } else {
            if self.right_nn.is_none() {
                return pre_output;
            }
            let mut right_nn = self.right_nn.as_ref().unwrap().clone();
            return right_nn.predict(input.clone());
        }
    }
}

impl NeuralNetwork for TrainableEitherNeuralNetwork {
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

        let pre_user_model_directory = append_dir(user_model_directory.clone(), "pre");
        self.pre_nn.save(pre_user_model_directory)?;
        match self.left_nn {
            Some(ref mut left_nn) => {
                let left_user_model_directory = append_dir(user_model_directory.clone(), "left");
                left_nn.save(left_user_model_directory)?;
            }
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => {
                let right_user_model_directory = append_dir(user_model_directory, "right");
                right_nn.save(right_user_model_directory)?;
            }
            None => {}
        }

        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }

    fn allocate(&mut self) {
        self.pre_nn.allocate();
        match self.left_nn {
            Some(ref mut left_nn) => left_nn.allocate(),
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => right_nn.allocate(),
            None => {}
        }
    }

    fn deallocate(&mut self) {
        self.pre_nn.deallocate();
        match self.left_nn {
            Some(ref mut left_nn) => left_nn.deallocate(),
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => right_nn.deallocate(),
            None => {}
        }
    }

    fn set_internal(&mut self) {
        self.model_directory = Directory::Internal(self.model_directory.path());
        self.pre_nn.set_internal();
        match self.left_nn {
            Some(ref mut left_nn) => left_nn.set_internal(),
            None => {}
        }
        match self.right_nn {
            Some(ref mut right_nn) => right_nn.set_internal(),
            None => {}
        }
    }

    fn duplicate(&self) -> WrappedNeuralNetwork {
        unimplemented!()
    }

    fn get_utils(&self) -> WrappedUtils {
        self.utils.clone()
    }
}

impl TrainableNeuralNetwork for TrainableEitherNeuralNetwork {
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
        let mut temp_neural_network =
            new_trainable_neural_network(NeuralNetworkCreationArguments::new(
                self.shape.clone(),
                None,
                None,
                append_dir(self.model_directory.path(), "temp"),
                self.utils.clone(),
            ));
        let temp_accuracy = temp_neural_network.train(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );

        let (left_inputs, left_targets): (Vec<Vec<f64>>, Vec<Vec<f64>>) = inputs
            .iter()
            .zip(targets.iter())
            .map(|(input, target)| {
                let prediction = temp_neural_network.predict(input.clone());
                (input, target, prediction)
            })
            .filter(|(input, target, prediction)| {
                // Check if the output matches the target
                let mut nb_correct_outputs = 0;
                for (o, t) in prediction.iter().zip(target.iter()) {
                    if (o - t).abs() < tolerance {
                        nb_correct_outputs += 1;
                    }
                }
                nb_correct_outputs == target.len()
            })
            .map(|(input, target, _)| (input.clone(), target.clone()))
            .unzip();

        let (right_inputs, right_targets): (Vec<Vec<f64>>, Vec<Vec<f64>>) = inputs
            .iter()
            .zip(targets.iter())
            .map(|(input, target)| {
                let prediction = temp_neural_network.predict(input.clone());
                (input, target, prediction)
            })
            .filter(|(input, target, prediction)| {
                // Check if the output matches the target
                let mut nb_correct_outputs = 0;
                for (o, t) in prediction.iter().zip(target.iter()) {
                    if (o - t).abs() < tolerance {
                        nb_correct_outputs += 1;
                    }
                }
                nb_correct_outputs != target.len()
            })
            .map(|(input, target, _)| (input.clone(), target.clone()))
            .unzip();

        // The number of failed predictions is below 100, so this instance is a leaf network
        let pre_model_directory = append_dir(self.model_directory.path(), "pre");
        if right_inputs.len() < 100 {
            self.pre_nn = temp_neural_network.clone();
            self.pre_nn
                .save(pre_model_directory.clone())
                .expect("Failed to save pre neural network");
            return temp_accuracy;
        }

        let (left_inputs_pre, left_targets_pre): (Vec<Vec<f64>>, Vec<Vec<f64>>) = left_inputs
            .iter()
            .map(|input| {
                let target = vec![1.0, 0.0];
                (input.clone(), target.clone())
            })
            .collect();
        let (right_inputs_pre, right_targets_pre): (Vec<Vec<f64>>, Vec<Vec<f64>>) = right_inputs
            .iter()
            .map(|input| {
                let target = vec![0.0, 1.0];
                (input.clone(), target.clone())
            })
            .collect();

        // conactenate all pre inputs
        let pre_inputs = [left_inputs_pre, right_inputs_pre].concat();
        // conactenate all pre targets
        let pre_targets = [left_targets_pre, right_targets_pre].concat();

        // Create the pre neural network
        let mut pre_nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            self.pre_shape.clone(),
            None,
            None,
            pre_model_directory,
            self.utils.clone(),
        ));

        // Train the pre neural network
        pre_nn.train(
            &pre_inputs,
            &pre_targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );

        self.pre_nn = pre_nn.clone();

        let left_model_directory = append_dir(self.model_directory.path(), "left");
        let mut left_nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            self.shape.clone(),
            None,
            Some(self.pre_shape.clone()),
            left_model_directory.clone(),
            self.utils.clone(),
        ));

        // Train the left neural network
        let left_accuracy = left_nn.train(
            &left_inputs,
            &left_targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );

        self.left_nn = Some(left_nn.clone());
        left_nn
            .save(left_model_directory.clone())
            .expect("Failed to save left neural network");

        // Train the right neural network
        let right_model_directory = append_dir(self.model_directory.path(), "right");
        let mut right_nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            self.shape.clone(),
            None,
            Some(self.pre_shape.clone()),
            right_model_directory.clone(),
            self.utils.clone(),
        ));

        // Train the right neural network
        let right_accuracy = right_nn.train(
            &right_inputs,
            &right_targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        );
        self.right_nn = Some(right_nn.clone());
        right_nn
            .save(right_model_directory.clone())
            .expect("Failed to save right neural network");

        left_accuracy + right_accuracy
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
        self.pre_nn.train_batch(
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
        let mut cloned_retry_nn =
            trainable_neural_network_from_disk(new_model_directory, self.utils.clone());
        cloned_retry_nn.set_internal();
        cloned_retry_nn
    }
}

impl Drop for TrainableEitherNeuralNetwork {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        // ensure that the model_directory exists
        if let Directory::User(_) = &self.model_directory {
            if std::fs::metadata(self.model_directory.path()).is_err() {
                std::fs::create_dir_all(self.model_directory.path()).unwrap();
            }
            self.deallocate();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::{
        nn::shape::{ActivationData, ActivationType, LayerShape},
        utilities::util::Utils,
    };

    #[test]
    fn test_retry_neural_network_train() {
        let utils = WrappedUtils::new(Utils::new(1000000000, 4));
        let mut nn = TrainableEitherNeuralNetwork::new(
            NeuralNetworkShape {
                layers: vec![
                    LayerShape {
                        layer_type: LayerType::Dense {
                            input_size: 3,
                            output_size: 3,
                        },
                        activation: ActivationData::new(ActivationType::ReLU),
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
            NeuralNetworkShape {
                layers: vec![LayerShape {
                    layer_type: LayerType::Dense {
                        input_size: 3,
                        output_size: 3,
                    },
                    activation: ActivationData::new_softmax(1.0),
                }],
            },
            "internal_model".to_string(),
            utils.clone(),
        );

        let input = vec![1.0, 1.0, 1.0];
        // put input 200 times in inputs
        let inputs = vec![input.clone(); 500];
        let target = vec![0.0, 0.0, 0.0];
        let targets = vec![target.clone(); 500];

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
