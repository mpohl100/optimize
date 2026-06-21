use matrix::persist::traits::PersistableMatrixTrait;
use matrix::persist::traits::PersistableValue;
use matrix::persist::wrapped::WrappedPersistableMatrix;

use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::mat::WrappedMatrix;

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

pub trait MatrixExtensionsPersistable<
    WeightT: PersistableValue + Default + Clone + From<f64> + 'static,
    BiasT: PersistableValue + Default + Clone + From<f64> + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &Box<dyn PersistableMatrixTrait<BiasT> + Send>,
    ) -> Vec<f64>;
}

pub trait MatrixExtensionsWrappedPersistable<
    WeightT: PersistableValue + Default + Clone + From<f64> + 'static,
    BiasT: PersistableValue + Default + Clone + From<f64> + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedPersistableMatrix<BiasT>,
    ) -> Vec<f64>;
}

pub trait TrainableMatrixExtensionsPersistable<
    WeightT: PersistableValue + Default + Clone + From<f64> + 'static,
    BiasT: PersistableValue + Default + Clone + From<f64> + 'static,
>: MatrixExtensionsPersistable<WeightT, BiasT>
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

pub trait TrainableMatrixExtensionsWrappedPersistable<
    WeightT: PersistableValue + Default + Clone + From<f64> + 'static,
    BiasT: PersistableValue + Default + Clone + From<f64> + 'static,
>: MatrixExtensionsWrappedPersistable<WeightT, BiasT>
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

pub trait MatrixExtensionsComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &CompositeMatrix<BiasT>,
    ) -> Vec<f64>;
}

pub trait TrainableMatrixExtensionsComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
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

pub trait MatrixExtensionsWrappedComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
>
{
    fn forward(
        &self,
        inputs: &[f64],
        biases: &WrappedCompositeMatrix<BiasT>,
    ) -> Vec<f64>;
}

pub trait TrainableMatrixExtensionsWrappedComposite<
    WeightT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Send + Sync + 'static,
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
