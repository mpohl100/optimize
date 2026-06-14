use alloc::allocatable::WrappedAllocatableTrait;
pub use matrix::ai_types::Bias;
pub use matrix::ai_types::BiasEntry;
pub use matrix::ai_types::NumberEntry;
pub use matrix::ai_types::Weight;
pub use matrix::ai_types::WeightEntry;
use matrix::persistable_matrix::PersistableMatrix;
use matrix::persistable_matrix::WrappedPersistableMatrix;

use super::super::traits::MatrixExtensionsPersistable;
use super::super::traits::MatrixExtensionsWrappedPersistable;
use super::super::traits::TrainableMatrixExtensionsPersistable;
use super::super::traits::TrainableMatrixExtensionsWrappedPersistable;
use crate::layer::matrix_extend::traits::MatrixExtensions;
use crate::layer::matrix_extend::traits::TrainableMatrixExtensions;

use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::mat::WrappedMatrix;

use matrix::persistable_matrix::PersistableValue;
use rayon::iter::ParallelIterator;

use num_traits::cast::NumCast;

impl MatrixExtensionsPersistable<NumberEntry, NumberEntry> for PersistableMatrix<NumberEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        self.mat().unwrap().forward(inputs, biases.mat().unwrap())
    }
}

impl MatrixExtensionsPersistable<WeightEntry, BiasEntry> for PersistableMatrix<WeightEntry> {
    fn forward(
        &self,
        inputs: &[f64],
        biases: &PersistableMatrix<BiasEntry>,
    ) -> Vec<f64> {
        self.mat().unwrap().forward(inputs, biases.mat().unwrap())
    }
}

impl MatrixExtensionsWrappedPersistable<NumberEntry, NumberEntry>
    for WrappedPersistableMatrix<NumberEntry>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        self.mat().lock().unwrap().forward(inputs, &biases.mat().lock().unwrap())
    }
}

impl MatrixExtensionsWrappedPersistable<WeightEntry, BiasEntry>
    for WrappedPersistableMatrix<WeightEntry>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedPersistableMatrix<BiasEntry>,
    ) -> Vec<f64> {
        self.mat().lock().unwrap().forward(inputs, &biases.mat().lock().unwrap())
    }
}

impl TrainableMatrixExtensionsPersistable<WeightEntry, BiasEntry>
    for PersistableMatrix<WeightEntry>
{
    fn backward_calculate_gradients(
        &self,
        d_out_vec: &[f64],
        input_cache: &[f64],
    ) {
        self.mat().unwrap().backward_calculate_gradients(d_out_vec, input_cache);
    }

    fn backward_calculate_weights_sec(
        &self,
        j: usize,
        d_out_vec_sec: &[f64],
    ) -> f64 {
        self.mat().unwrap().backward_calculate_weights_sec(j, d_out_vec_sec)
    }

    fn update_weights(
        &self,
        learning_rate: f64,
    ) {
        self.mat().unwrap().update_weights(learning_rate);
    }

    fn adjust_adam(
        &self,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        t: usize,
        learning_rate: f64,
    ) {
        self.mat().unwrap().adjust_adam(beta1, beta2, epsilon, t, learning_rate);
    }
}

impl TrainableMatrixExtensionsWrappedPersistable<WeightEntry, BiasEntry>
    for WrappedPersistableMatrix<WeightEntry>
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
