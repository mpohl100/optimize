use crate::neural::nn::directory::Directory;
use crate::neural::nn::shape::NeuralNetworkShape;

pub trait NeuralNetwork: Clone {
    fn predict(&mut self, input: Vec<f64>) -> Vec<f64>;
    /// Creates a new `NeuralNetwork` from the given model directory.
    fn from_disk(model_directory: String) -> Option<Self>;

    /// Performs a forward pass through the network with the given input.
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Performs a forward pass through the network with the given input doing batch caching.
    fn forward_batch(&mut self, input: &[f64]) -> Vec<f64>;
    fn shape(&self) -> &NeuralNetworkShape;
    fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>>;
    fn get_model_directory(&self) -> Directory;
}

pub trait TrainableNeuralNetwork: NeuralNetwork {
    /// Performs a backward pass through the network with the given output gradient.
    fn backward(&mut self, grad_output: Vec<f64>);

    /// Performs a backward pass through the network with the given output gradient doing batch caching.
    fn backward_batch(&mut self, grad_output: Vec<f64>);

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
}
