use crate::neural::nn::shape::NeuralNetworkShape;

#[derive(Clone)]
pub struct TrainingParams {
    shape: NeuralNetworkShape,
    num_training_samples: usize,
    num_verification_samples: usize,
    learning_rate: f64,
    epochs: usize,
    tolerance: f64,
}

impl TrainingParams {
    pub fn new(
        shape: NeuralNetworkShape,
        num_training_samples: usize,
        num_verification_samples: usize,
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            shape,
            num_training_samples,
            num_verification_samples,
            learning_rate,
            epochs,
            tolerance,
        }
    }

    pub fn shape(&self) -> &NeuralNetworkShape {
        &self.shape
    }

    pub fn num_training_samples(&self) -> usize {
        self.num_training_samples
    }

    pub fn num_verification_samples(&self) -> usize {
        self.num_verification_samples
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
