use super::dense_layer::Weight;

use matrix::mat::Matrix;
use matrix::mat::WrappedMatrix;

use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
pub trait MatrixExtensions<T: Default + Clone> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &[f64],
    ) -> Vec<f64>;
}

impl MatrixExtensions<f64> for WrappedMatrix<f64> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &[f64],
    ) -> Vec<f64> {
        self.mat()
            .lock()
            .unwrap()
            .par_indexed_iter()
            .map(|(row_idx, weights_row)| {
                weights_row.iter().zip(inputs.iter()).map(|(&w, &x)| w * x).sum::<f64>()
                    + biases[row_idx] // Use the bias corresponding to the row index
            })
            .collect::<Vec<f64>>()
    }
}

impl MatrixExtensions<Weight> for WrappedMatrix<Weight> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &[f64],
    ) -> Vec<f64> {
        self.mat()
            .lock()
            .unwrap()
            .par_indexed_iter()
            .map(|(row_idx, weights_row)| {
                let value =
                    weights_row.iter().zip(inputs.iter()).map(|(w, &x)| w.value * x).sum::<f64>()
                        + biases[row_idx]; // Use the bias corresponding to the row index
                value
            })
            .collect::<Vec<f64>>()
    }
}

pub trait TrainableMatrixExtensions<T: Default + Clone>: MatrixExtensions<T> {
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &Vec<f64>,
        input_cache: &Vec<f64>,
    );
    fn backward_calculate_weights_sec(&mut self) -> Vec<T>;
    fn update_weights(
        &mut self,
        learning_rate: f64,
    );
    fn adjust_adam(
        &mut self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    );
}

impl TrainableMatrixExtensions<Weight> for WrappedMatrix<Weight> {
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &Vec<f64>,
        input_cache: &Vec<f64>,
    ) {
        self.mat().lock().unwrap().par_indexed_iter_mut().for_each(|(i, row_grad)| {
            row_grad.iter_mut().enumerate().for_each(|(j, grad)| {
                grad.grad = d_out_vec[i] * input_cache[j];
            });
        });
    }

    fn backward_calculate_weights_sec(&mut self) -> Vec<f64> {
        // Implement secondary weight calculation here
        vec![]
    }

    fn update_weights(
        &mut self,
        learning_rate: f64,
    ) {
        // Implement weight update logic here
    }

    fn adjust_adam(
        &mut self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    ) {
        // Implement Adam adjustment logic here
    }
}
