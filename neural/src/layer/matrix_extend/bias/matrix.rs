use super::super::traits::MatrixExtensions;
use super::super::traits::TrainableMatrixExtensions;
pub use matrix::ai_types::BiasEntry;

use matrix::mat::WrappedMatrix;

use rayon::iter::ParallelIterator;

use num_traits::cast::NumCast;

impl MatrixExtensions<BiasEntry, BiasEntry> for WrappedMatrix<BiasEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        self.mat()
            .lock()
            .unwrap()
            .par_indexed_iter()
            .map(|(row_idx, weights_row)| {
                let value =
                    weights_row.iter().zip(inputs.iter()).map(|(w, &x)| w.0.value * x).sum::<f64>()
                        + biases.mat().lock().unwrap().get_mut_unchecked(row_idx, 0).0.value; // Use the bias corresponding to the row index
                value
            })
            .collect::<Vec<f64>>()
    }
}

impl TrainableMatrixExtensions<BiasEntry, BiasEntry> for WrappedMatrix<BiasEntry> {
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    ) {
        self.mat().lock().unwrap().par_indexed_iter_mut().for_each(|(i, row_grad)| {
            row_grad.iter_mut().enumerate().for_each(|(j, grad)| {
                grad.0.grad = d_out_vec[i] * input_cache[j];
            });
        });
    }

    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64 {
        self.mat()
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, row)| row[j].0.value * d_out_vec_sec[i])
            .sum::<f64>()
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.mat().lock().unwrap().par_iter_mut().for_each(|weights_row| {
            for weight in weights_row.iter_mut() {
                weight.0.value = learning_rate.mul_add(-weight.0.grad, weight.0.value);
            }
        });
    }

    fn adjust_adam(
        &self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    ) {
        let t_f: f64 = NumCast::from(t).expect("Failed to convert time step to f64");
        let beta1_pow_t = beta1.powf(t_f);
        let beta2_pow_t = beta2.powf(t_f);
        self.mat().lock().unwrap().par_iter_mut().for_each(|weight_row| {
            for weight in weight_row.iter_mut() {
                let grad = weight.0.grad;

                // Update first and second moments
                weight.0.m = beta1.mul_add(weight.0.m, (1.0 - beta1) * grad);
                weight.0.v = beta2.mul_add(weight.0.v, (1.0 - beta2) * grad.powi(2));

                // Bias correction
                let m_hat = weight.0.m / (1.0 - beta1_pow_t);
                let v_hat = weight.0.v / (1.0 - beta2_pow_t);

                // Adjusted learning rate and update
                let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);
                weight.0.value -= adjusted_learning_rate * m_hat;
            }
        });
    }
}
