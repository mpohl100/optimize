pub use matrix::ai_types::Bias;
pub use matrix::ai_types::BiasEntry;
pub use matrix::ai_types::NumberEntry;
pub use matrix::ai_types::Weight;
pub use matrix::ai_types::WeightEntry;

use alloc::allocatable::Allocatable;

use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::mat::WrappedMatrix;

use matrix::persistable_matrix::PersistableValue;
use rayon::iter::ParallelIterator;

use num_traits::cast::NumCast;

pub trait MatrixExtensions<
    WeightT: Default + Clone + From<f64> + 'static,
    BiasT: Default + Clone + From<f64> + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedMatrix<BiasT>,
    ) -> Vec<f64>;
}

impl MatrixExtensions<f64, f64> for WrappedMatrix<f64> {
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
                let value = weights_row.iter().zip(inputs.iter()).map(|(w, &x)| w * x).sum::<f64>()
                    + *biases.mat().lock().unwrap().get_mut_unchecked(row_idx, 0); // Use the bias corresponding to the row index
                value
            })
            .collect::<Vec<f64>>()
    }
}

impl MatrixExtensions<Weight, Bias> for WrappedMatrix<Weight> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedMatrix<Bias>,
    ) -> Vec<f64> {
        self.mat()
            .lock()
            .unwrap()
            .par_indexed_iter()
            .map(|(row_idx, weights_row)| {
                let value =
                    weights_row.iter().zip(inputs.iter()).map(|(w, &x)| w.value * x).sum::<f64>()
                        + biases.mat().lock().unwrap().get_mut_unchecked(row_idx, 0).value; // Use the bias corresponding to the row index
                value
            })
            .collect::<Vec<f64>>()
    }
}

pub trait TrainableMatrixExtensions<
    WeightT: Default + Clone + From<f64> + 'static,
    BiasT: Default + Clone + From<f64> + 'static,
>: MatrixExtensions<WeightT, BiasT>
{
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    );
    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64;
    fn update_weights(
        &self,
        learning_rate: f64,
    );
    fn adjust_adam(
        &self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    );
}

impl TrainableMatrixExtensions<Weight, Bias> for WrappedMatrix<Weight> {
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    ) {
        self.mat().lock().unwrap().par_indexed_iter_mut().for_each(|(i, row_grad)| {
            row_grad.iter_mut().enumerate().for_each(|(j, grad)| {
                grad.grad = d_out_vec[i] * input_cache[j];
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
            .map(|(i, row)| row[j].value * d_out_vec_sec[i])
            .sum::<f64>()
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.mat().lock().unwrap().par_iter_mut().for_each(|weights_row| {
            for weight in weights_row.iter_mut() {
                weight.value -= learning_rate * weight.grad;
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
                let grad = weight.grad;

                // Update first and second moments
                weight.m = beta1.mul_add(weight.m, (1.0 - beta1) * grad);
                weight.v = beta2.mul_add(weight.v, (1.0 - beta2) * grad.powi(2));

                // Bias correction
                let m_hat = weight.m / (1.0 - beta1_pow_t);
                let v_hat = weight.v / (1.0 - beta2_pow_t);

                // Adjusted learning rate and update
                let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);
                weight.value -= adjusted_learning_rate * m_hat;
            }
        });
    }
}

impl MatrixExtensions<NumberEntry, NumberEntry> for WrappedMatrix<NumberEntry> {
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
                weights_row.iter().zip(inputs.iter()).map(|(w, x)| w.0 * x).sum::<f64>()
                    + biases.mat().lock().unwrap().get_mut_unchecked(row_idx, 0).0
                // Use the bias corresponding to the row index
            })
            .collect::<Vec<f64>>()
    }
}

impl MatrixExtensions<WeightEntry, BiasEntry> for WrappedMatrix<WeightEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedMatrix<BiasEntry>,
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

impl TrainableMatrixExtensions<WeightEntry, BiasEntry> for WrappedMatrix<WeightEntry> {
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
                weight.0.value -= learning_rate * weight.0.grad;
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
                weight.0.value -= learning_rate * weight.0.grad;
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

pub trait MatrixExtensionsComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &CompositeMatrix<BiasT>,
    ) -> Vec<f64>;
}

impl MatrixExtensionsComposite<NumberEntry, NumberEntry> for CompositeMatrix<NumberEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        forward_impl(self, inputs, biases, |w| w.0, |b| b.0)
    }
}

impl MatrixExtensionsComposite<WeightEntry, BiasEntry> for CompositeMatrix<WeightEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &CompositeMatrix<BiasEntry>,
    ) -> Vec<f64> {
        forward_impl(self, inputs, biases, |w| w.0.value, |b| b.0.value)
    }
}

impl MatrixExtensionsComposite<BiasEntry, BiasEntry> for CompositeMatrix<BiasEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        forward_impl(self, inputs, biases, |w| w.0.value, |b| b.0.value)
    }
}

pub trait TrainableMatrixExtensionsComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + 'static,
>: MatrixExtensionsComposite<WeightT, BiasT>
{
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    );
    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64;
    fn update_weights(
        &self,
        learning_rate: f64,
    );
    fn adjust_adam(
        &self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    );
}

impl TrainableMatrixExtensionsComposite<WeightEntry, BiasEntry> for CompositeMatrix<WeightEntry> {
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
                matrix.allocate();
                let d_out_row_start = i * self.get_slice_num_rows();
                let d_out_row_end = d_out_row_start + matrix.rows();
                let input_cache_row_start = j * self.get_slice_num_cols();
                let input_cache_row_end = input_cache_row_start + matrix.cols();
                matrix.mat().unwrap().backward_calculate_gradients(
                    &d_out_vec[d_out_row_start..d_out_row_end],
                    &input_cache[input_cache_row_start..input_cache_row_end],
                );
            });
        });
    }

    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64 {
        let mut result = 0.0;
        self.matrices().mat().lock().unwrap().iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(k, matrix)| {
                if k == j {
                    matrix.allocate();
                    let row_start = i * self.get_slice_num_rows();
                    let row_end = row_start + matrix.rows();
                    result += matrix
                        .mat()
                        .unwrap()
                        .backward_calculate_weights_sec(j, &d_out_vec_sec[row_start..row_end]);
                }
            });
        });
        result
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.matrices().mat().lock().unwrap().iter_mut().for_each(|row| {
            for matrix in row.iter_mut() {
                matrix.allocate();
                matrix.mat().unwrap().update_weights(learning_rate);
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
                matrix.allocate();
                matrix.mat().unwrap().adjust_adam(beta1, beta2, epsilon, t, learning_rate);
            }
        });
    }
}

impl TrainableMatrixExtensionsComposite<BiasEntry, BiasEntry> for CompositeMatrix<BiasEntry> {
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    ) {
        self.matrices().mat().lock().unwrap().iter_mut().enumerate().for_each(|(i, row)| {
            for matrix in row.iter_mut() {
                matrix.allocate();
                let row_start = i * self.get_slice_num_rows();
                let row_end = row_start + matrix.rows();
                matrix.mat().unwrap().backward_calculate_gradients(
                    &d_out_vec[row_start..row_end],
                    &input_cache[row_start..row_end],
                );
            }
        });
    }

    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64 {
        let mut result = 0.0;
        self.matrices().mat().lock().unwrap().iter_mut().for_each(|row| {
            row.iter_mut().enumerate().for_each(|(k, matrix)| {
                if k == j {
                    matrix.allocate();
                    let row_start = j * self.get_slice_num_rows();
                    let row_end = row_start + matrix.rows();
                    result += matrix
                        .mat()
                        .unwrap()
                        .backward_calculate_weights_sec(j, &d_out_vec_sec[row_start..row_end]);
                }
            });
        });
        result
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.matrices().mat().lock().unwrap().iter_mut().for_each(|row| {
            for matrix in row.iter_mut() {
                matrix.allocate();
                matrix.mat().unwrap().update_weights(learning_rate);
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
                matrix.allocate();
                matrix.mat().unwrap().adjust_adam(beta1, beta2, epsilon, t, learning_rate);
            }
        });
    }
}

pub trait MatrixExtensionsWrappedComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedCompositeMatrix<BiasT>,
    ) -> Vec<f64>;
}

impl MatrixExtensionsWrappedComposite<NumberEntry, NumberEntry>
    for WrappedCompositeMatrix<NumberEntry>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        self.mat().lock().unwrap().forward(inputs, &biases.mat().lock().unwrap())
    }
}

pub trait TrainableMatrixExtensionsWrappedComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + 'static,
>: MatrixExtensionsWrappedComposite<WeightT, BiasT>
{
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    );
    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64;
    fn update_weights(
        &self,
        learning_rate: f64,
    );
    fn adjust_adam(
        &self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    );
}

impl MatrixExtensionsWrappedComposite<WeightEntry, BiasEntry>
    for WrappedCompositeMatrix<WeightEntry>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedCompositeMatrix<BiasEntry>,
    ) -> Vec<f64> {
        self.mat().lock().unwrap().forward(inputs, &biases.mat().lock().unwrap())
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

impl TrainableMatrixExtensionsWrappedComposite<WeightEntry, BiasEntry>
    for WrappedCompositeMatrix<WeightEntry>
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

fn forward_impl<
    WeightT: Default + Clone + PersistableValue + From<f64> + Sync + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + Sync + 'static,
>(
    mat: &CompositeMatrix<WeightT>,
    inputs: &[f64],
    biases: &CompositeMatrix<BiasT>,
    weight_value_retriever: fn(&WeightT) -> f64,
    bias_value_retriever: fn(&BiasT) -> f64,
) -> Vec<f64> {
    let mut outputs = vec![0.0; mat.rows()];
    mat.matrices().mat().lock().unwrap().iter_mut().enumerate().for_each(|(i, row)| {
        let num_rows = if (i + 1) * mat.get_slice_num_rows() > mat.rows() {
            mat.rows() - i * mat.get_slice_num_rows()
        } else {
            mat.get_slice_num_rows()
        };
        let mut local_outputs = vec![0.0; num_rows];
        let local_inputs =
            &inputs[i * mat.get_slice_num_rows()..i * mat.get_slice_num_rows() + num_rows];
        for matrix in row.iter_mut() {
            matrix.allocate();
            let col_outputs = matrix
                .mat()
                .unwrap()
                .mat()
                .lock()
                .unwrap()
                .par_iter()
                .map(|row| {
                    let value = row
                        .iter()
                        .zip(local_inputs.iter())
                        .map(|(w, &x)| weight_value_retriever(w) * x)
                        .sum::<f64>();
                    value
                })
                .collect::<Vec<f64>>();
            for (k, val) in col_outputs.iter().enumerate() {
                local_outputs[k] += val;
            }
        }
        // add local biases
        let mut local_biases = biases.matrices().get_unchecked(i, 0);
        local_biases.allocate();
        local_biases.mat().unwrap().mat().lock().unwrap().iter_mut().enumerate().for_each(
            |(bias_index, bias_row)| {
                for bias in bias_row.iter_mut() {
                    local_outputs[bias_index] += bias_value_retriever(bias);
                }
            },
        );

        for (k, val) in local_outputs.iter().enumerate() {
            outputs[i * mat.get_slice_num_rows() + k] += val;
        }
    });
    outputs
}
