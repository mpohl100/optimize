use crate::neural::nn::shape::NeuralNetworkShape;

#[derive(Clone)]
pub struct TrainingParams {
    shape: NeuralNetworkShape,
    training_verification_ratio: f64,
    learning_rate: f64,
    epochs: usize,
    tolerance: f64,
}

impl TrainingParams {
    pub fn new(
        shape: NeuralNetworkShape,
        training_verification_ratio: f64,
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            shape,
            training_verification_ratio,
            learning_rate,
            epochs,
            tolerance,
        }
    }

    pub fn shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    pub fn num_training_samples(&self) -> f64 {
        self.training_verification_ratio 
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn set_shape(&mut self, shape: NeuralNetworkShape) {
        self.shape = shape;
    }
}
