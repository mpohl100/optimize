use matrix::ai_types::BiasEntry;
use matrix::ai_types::NumberEntry;
use matrix::ai_types::WeightEntry;
use matrix::persist::traits::PersistableMatrixTrait;
use matrix::persist::wrapped::WrappedPersistableMatrix;

use super::super::traits::MatrixExtensionsPersistable;
use super::super::traits::MatrixExtensionsWrappedPersistable;
use super::super::traits::TrainableMatrixExtensionsPersistable;
use super::super::traits::TrainableMatrixExtensionsWrappedPersistable;
use crate::layer::matrix_extend::traits::MatrixExtensions;
use crate::layer::matrix_extend::traits::TrainableMatrixExtensions;

impl MatrixExtensionsPersistable<NumberEntry, NumberEntry>
    for Box<dyn PersistableMatrixTrait<NumberEntry> + Send>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Self,
    ) -> Vec<f64> {
        self.mat().unwrap().forward(inputs, biases.mat().unwrap())
    }
}

impl MatrixExtensionsPersistable<WeightEntry, BiasEntry>
    for Box<dyn PersistableMatrixTrait<WeightEntry> + Send>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Box<dyn PersistableMatrixTrait<BiasEntry> + Send>,
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
    for Box<dyn PersistableMatrixTrait<WeightEntry> + Send>
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
