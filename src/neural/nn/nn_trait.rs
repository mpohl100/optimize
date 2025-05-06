use crate::neural::nn::directory::Directory;
use crate::neural::nn::shape::NeuralNetworkShape;
use std::sync::{Arc, Mutex};

pub trait NeuralNetwork: std::fmt::Debug {
    fn predict(&mut self, input: Vec<f64>) -> Vec<f64>;
    fn shape(&self) -> NeuralNetworkShape;
    fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>>;
    fn get_model_directory(&self) -> Directory;
    fn allocate(&mut self);
    fn deallocate(&mut self);
    fn set_internal(&mut self);
    fn duplicate(&self) -> WrappedNeuralNetwork;
}

#[derive(Debug, Clone)]
pub struct WrappedNeuralNetwork {
    nn: Arc<Mutex<Box<dyn NeuralNetwork + Send>>>,
}

impl WrappedNeuralNetwork {
    pub fn new(nn: Box<dyn NeuralNetwork + Send>) -> Self {
        Self {
            nn: Arc::new(Mutex::new(nn)),
        }
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.nn.lock().unwrap().predict(input)
    }

    pub fn shape(&self) -> NeuralNetworkShape {
        self.nn.lock().unwrap().shape()
    }

    pub fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        self.nn.lock().unwrap().save(user_model_directory)
    }

    pub fn allocate(&self) {
        self.nn.lock().unwrap().allocate();
    }

    pub fn deallocate(&self) {
        self.nn.lock().unwrap().deallocate();
    }

    pub fn set_internal(&mut self) {
        self.nn.lock().unwrap().set_internal();
    }

    pub fn duplicate(&self) -> WrappedNeuralNetwork {
        self.nn.lock().unwrap().duplicate()
    }
}

pub trait TrainableNeuralNetwork: NeuralNetwork {
    /// Trains the neural network using the given inputs, targets, learning rate, and number of epochs.
    /// Includes validation using a split of the data.
    #[allow(clippy::too_many_arguments)]
    fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
    );

    /// Trains the neural network doing batch back propagation.
    fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    );

    /// Returns the input size of the first layer in the network.
    fn input_size(&self) -> usize;

    /// Returns the output size of the last layer in the network.
    fn output_size(&self) -> usize;

    fn duplicate_trainable(&self) -> WrappedTrainableNeuralNetwork;
}

#[derive(Debug, Clone)]
pub struct WrappedTrainableNeuralNetwork {
    nn: Arc<Mutex<Box<dyn TrainableNeuralNetwork + Send>>>,
}

impl WrappedTrainableNeuralNetwork {
    pub fn new(nn: Box<dyn TrainableNeuralNetwork + Send>) -> Self {
        Self {
            nn: Arc::new(Mutex::new(nn)),
        }
    }

    pub fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.nn.lock().unwrap().predict(input)
    }

    pub fn shape(&self) -> NeuralNetworkShape {
        self.nn.lock().unwrap().shape()
    }

    pub fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        self.nn.lock().unwrap().save(user_model_directory)
    }

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
        self.nn.lock().unwrap().train(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        batch_size: usize,
    ) {
        self.nn.lock().unwrap().train_batch(
            inputs,
            targets,
            learning_rate,
            epochs,
            tolerance,
            batch_size,
        )
    }

    pub fn input_size(&self) -> usize {
        self.nn.lock().unwrap().input_size()
    }

    pub fn output_size(&self) -> usize {
        self.nn.lock().unwrap().output_size()
    }

    pub fn get_model_directory(&self) -> Directory {
        self.nn.lock().unwrap().get_model_directory()
    }

    pub fn allocate(&self) {
        self.nn.lock().unwrap().allocate();
    }

    pub fn deallocate(&self) {
        self.nn.lock().unwrap().deallocate();
    }

    pub fn set_internal(&mut self) {
        self.nn.lock().unwrap().set_internal();
    }

    pub fn duplicate_trainable(&self) -> WrappedTrainableNeuralNetwork {
        self.nn.lock().unwrap().duplicate_trainable()
    }
}
