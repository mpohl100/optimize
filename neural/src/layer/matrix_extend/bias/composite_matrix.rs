use alloc::allocatable::WrappedAllocatableTrait;
use matrix::ai_types::BiasEntry;

use crate::layer::matrix_extend::composite_impl::forward_impl;
use crate::layer::matrix_extend::traits::MatrixExtensionsComposite;
use crate::layer::matrix_extend::traits::MatrixExtensionsWrappedComposite;
use crate::layer::matrix_extend::traits::TrainableMatrixExtensionsComposite;
use crate::layer::matrix_extend::traits::TrainableMatrixExtensionsWrappedComposite;
use crate::layer::matrix_extend::traits::TrainableMatrixExtensionsWrappedPersistable;

use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;

impl MatrixExtensionsComposite<BiasEntry, BiasEntry> for CompositeMatrix<BiasEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        forward_impl(self, inputs, biases, |w| w.0.value, |b| b.0.value)
    }
}

impl TrainableMatrixExtensionsComposite<BiasEntry, BiasEntry> for CompositeMatrix<BiasEntry> {
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    ) {
        /*
        self.mat().lock().unwrap().par_indexed_iter_mut().for_each(|(i, row_grad)| {
            row_grad.iter_mut().enumerate().for_each(|(j, grad)| {
                grad.0.grad = d_out_vec[i] * input_cache[j];
            });
        });
        */
        self.matrices().mat().lock().unwrap().iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(j, matrix)| {
                let mut alloc_manager = self.get_alloc_manager();
                alloc_manager.allocate(matrix);
                matrix.mark_for_use();
                let d_out_row_start = i * self.get_slice_num_rows();
                let d_out_row_end = d_out_row_start + matrix.rows();
                let input_cache_row_start = j * self.get_slice_num_cols();
                let input_cache_row_end = input_cache_row_start + matrix.cols();
                matrix.backward_calculate_gradients(
                    &d_out_vec[d_out_row_start..d_out_row_end],
                    &input_cache[input_cache_row_start..input_cache_row_end],
                );
                matrix.free_from_use();
            });
        });
    }

    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64 {
        /*
            self.mat()
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, row)| row[j].0.value * d_out_vec_sec[i])
            .sum::<f64>()
        */
        let mut result = 0.0;
        // sum the jth column across all sub-matrices
        let num_cols_per_matrix = self.get_slice_num_cols();
        let persistable_col = j / num_cols_per_matrix;
        let local_col_index = j % num_cols_per_matrix;
        let binding = self.matrices().mat();
        let mut matrices_guard = binding.lock().unwrap();
        for i in 0..matrices_guard.rows() {
            let matrix = matrices_guard.get_mut_unchecked(i, persistable_col);

            let mut alloc_manager = self.get_alloc_manager();
            alloc_manager.allocate(matrix);
            matrix.mark_for_use();
            let row_start = i * self.get_slice_num_rows();
            let row_end = row_start + matrix.rows();
            result += matrix.backward_calculate_weights_sec(
                local_col_index,
                &d_out_vec_sec[row_start..row_end],
            );
            matrix.free_from_use();
        }
        drop(matrices_guard);
        result
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.matrices().mat().lock().unwrap().iter_mut().for_each(|row| {
            for matrix in row.iter_mut() {
                let mut alloc_manager = self.get_alloc_manager();
                alloc_manager.allocate(matrix);
                matrix.mark_for_use();
                matrix.update_weights(learning_rate);
                matrix.free_from_use();
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
        self.matrices().mat().lock().unwrap().iter_mut().for_each(|row| {
            for matrix in row.iter_mut() {
                let mut alloc_manager = self.get_alloc_manager();
                alloc_manager.allocate(matrix);
                matrix.mark_for_use();
                matrix.adjust_adam(beta1, beta2, epsilon, t, learning_rate);
                matrix.free_from_use();
            }
        });
    }
}

impl MatrixExtensionsWrappedComposite<BiasEntry, BiasEntry> for WrappedCompositeMatrix<BiasEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        self.mat().lock().unwrap().forward(inputs, &biases.mat().lock().unwrap())
    }
}

impl TrainableMatrixExtensionsWrappedComposite<BiasEntry, BiasEntry>
    for WrappedCompositeMatrix<BiasEntry>
{
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    ) {
        self.mat().lock().unwrap().backward_calculate_gradients(d_out_vec, input_cache);
    }

    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64 {
        self.mat().lock().unwrap().backward_calculate_weights_sec(j, d_out_vec_sec)
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.mat().lock().unwrap().update_weights(learning_rate);
    }

    fn adjust_adam(
        &self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    ) {
        self.mat().lock().unwrap().adjust_adam(beta1, beta2, epsilon, t, learning_rate);
    }
}
