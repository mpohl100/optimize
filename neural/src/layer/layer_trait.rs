use crate::utilities::util::WrappedUtils;
use matrix::ai_types::BiasEntry;
use matrix::ai_types::NumberEntry;
use matrix::ai_types::WeightEntry;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::persistable_matrix::PersistableValue;
use utils::safer::safe_lock;

use std::error::Error;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
// A trait representing a layer in a neural network.
/// Provides methods for the forward pass, backward pass, weight updates, and layer size information.
pub trait Layer<
    WeightT: Debug + Clone + PersistableValue + From<f64> + 'static,
    BiasT: Debug + Clone + PersistableValue + From<f64> + 'static,
>: Debug
{
    /// Performs the forward pass of the layer, computing the output based on the input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to a vector of `f64` values representing the input data.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values representing the output of the layer.
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64>;

    /// Performs the forward pass of the layer for inputs doing batch caching.
    fn forward_batch(
        &mut self,
        input: &[f64],
    ) -> Vec<f64>;

    /// Returns the input size of the layer.
    ///
    /// # Returns
    ///
    /// * A `usize` value representing the number of input neurons.
    fn input_size(&self) -> usize;

    /// Returns the output size of the layer.
    ///
    /// # Returns
    ///
    /// * A `usize` value representing the number of output neurons.
    fn output_size(&self) -> usize;

    /// Saves the layer to a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer could not be saved to the specified path.
    fn save(
        &self,
        path: String,
    ) -> Result<(), Box<dyn Error>>;

    /// Reads the layer from a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer could not be read from the specified path.
    fn read(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>>;

    /// Returns the weights of the layer.
    fn get_weights(&self) -> WrappedCompositeMatrix<WeightT>;

    /// Returns the biases of the layer.
    fn get_biases(&self) -> WrappedCompositeMatrix<BiasT>;

    /// Marks the layer as in use.
    fn cleanup(&self);

    /// Assigns the weight of the input other layer
    fn assign_weights(
        &mut self,
        other: WrappedLayer<NumberEntry, NumberEntry>,
    );

    /// Assigns the weight of the input other layer
    fn assign_trainable_weights(
        &mut self,
        other: WrappedTrainableLayer<WeightEntry, BiasEntry>,
    );
}

#[derive(Debug, Clone)]
pub struct WrappedLayer<
    WeightT: Debug + Clone + PersistableValue,
    BiasT: Debug + Clone + PersistableValue,
> {
    layer: Arc<Mutex<Box<dyn Layer<WeightT, BiasT> + Send>>>,
}

impl<
        WeightT: Debug + Clone + PersistableValue + From<f64> + 'static,
        BiasT: Debug + Clone + PersistableValue + From<f64> + 'static,
    > WrappedLayer<WeightT, BiasT>
{
    #[must_use]
    pub fn new(layer: Box<dyn Layer<WeightT, BiasT> + Send>) -> Self {
        Self { layer: Arc::new(Mutex::new(layer)) }
    }

    pub fn cleanup(&self) {
        safe_lock(&self.layer).cleanup();
    }

    pub fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        safe_lock(&self.layer).forward(input, utils)
    }

    pub fn forward_batch(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        safe_lock(&self.layer).forward_batch(input)
    }

    #[must_use]
    pub fn input_size(&self) -> usize {
        safe_lock(&self.layer).input_size()
    }

    #[must_use]
    pub fn output_size(&self) -> usize {
        safe_lock(&self.layer).output_size()
    }

    /// Saves the layer to a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer could not be saved to the specified path.
    pub fn save(
        &self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        safe_lock(&self.layer).save(path)
    }

    /// Reads the layer from a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer could not be read from the specified path.
    pub fn read(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        safe_lock(&self.layer).read(path)
    }

    #[must_use]
    pub fn get_weights(&self) -> WrappedCompositeMatrix<WeightT> {
        safe_lock(&self.layer).get_weights()
    }

    #[must_use]
    pub fn get_biases(&self) -> WrappedCompositeMatrix<BiasT> {
        safe_lock(&self.layer).get_biases()
    }

    pub fn assign_weights(
        &mut self,
        other: &WrappedLayer<NumberEntry, NumberEntry>,
    ) {
        safe_lock(&self.layer).assign_weights(other.clone());
    }

    pub fn assign_trainable_weights(
        &mut self,
        other: &WrappedTrainableLayer<WeightEntry, BiasEntry>,
    ) {
        safe_lock(&self.layer).assign_trainable_weights(other.clone());
    }
}
pub trait TrainableLayer<
    WeightT: Debug + Clone + PersistableValue + From<f64> + 'static,
    BiasT: Debug + Clone + PersistableValue + From<f64> + 'static,
>: Layer<WeightT, BiasT>
{
    /// Performs the backward pass of the layer, computing the gradient based on the output gradient.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - A reference to a vector of `f64` values representing the gradient of the loss
    ///   with respect to the output of this layer.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values representing the gradient of the loss with respect to the input.
    fn backward(
        &mut self,
        grad_output: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64>;

    /// Performs the backward pass of the layer for inputs doing batch caching.
    fn backward_batch(
        &mut self,
        grad_output: &[f64],
    ) -> Vec<f64>;

    /// Updates the weights of the layer based on the specified learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - A `f64` value representing the learning rate for weight updates.
    fn update_weights(
        &mut self,
        learning_rate: f64,
        utils: WrappedUtils,
    );

    /// Adjusts the weights according to the Adam optimizer.
    fn adjust_adam(
        &mut self,
        t: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        utils: WrappedUtils,
    );

    /// Saves the layer to a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer weights could not be saved to the specified path.
    fn save_weight(
        &self,
        path: String,
    ) -> Result<(), Box<dyn Error>>;

    /// Reads the layer from a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer weights could not be read from the specified path.
    fn read_weight(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>>;
}

#[derive(Debug, Clone)]
pub struct WrappedTrainableLayer<
    WeightT: Debug + Clone + PersistableValue,
    BiasT: Debug + Clone + PersistableValue,
> {
    layer: Arc<Mutex<Box<dyn TrainableLayer<WeightT, BiasT> + Send>>>,
}

impl<
        WeightT: Debug + Clone + PersistableValue + From<f64> + 'static,
        BiasT: Debug + Clone + PersistableValue + From<f64> + 'static,
    > WrappedTrainableLayer<WeightT, BiasT>
{
    #[must_use]
    pub fn new(layer: Box<dyn TrainableLayer<WeightT, BiasT> + Send>) -> Self {
        Self { layer: Arc::new(Mutex::new(layer)) }
    }

    pub fn cleanup(&self) {
        safe_lock(&self.layer).cleanup();
    }

    pub fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        safe_lock(&self.layer).forward(input, utils)
    }

    pub fn forward_batch(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        safe_lock(&self.layer).forward_batch(input)
    }

    #[must_use]
    pub fn input_size(&self) -> usize {
        safe_lock(&self.layer).input_size()
    }

    #[must_use]
    pub fn output_size(&self) -> usize {
        safe_lock(&self.layer).output_size()
    }

    /// Reads the layer from a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer could not be read from the specified path.
    pub fn read(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        safe_lock(&self.layer).read(path)
    }

    #[must_use]
    pub fn get_weights(&self) -> WrappedCompositeMatrix<WeightT> {
        safe_lock(&self.layer).get_weights()
    }

    #[must_use]
    pub fn get_biases(&self) -> WrappedCompositeMatrix<BiasT> {
        safe_lock(&self.layer).get_biases()
    }

    pub fn backward(
        &mut self,
        grad_output: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        safe_lock(&self.layer).backward(grad_output, utils)
    }

    pub fn backward_batch(
        &mut self,
        grad_output: &[f64],
    ) -> Vec<f64> {
        safe_lock(&self.layer).backward_batch(grad_output)
    }

    pub fn update_weights(
        &mut self,
        learning_rate: f64,
        utils: WrappedUtils,
    ) {
        safe_lock(&self.layer).update_weights(learning_rate, utils);
    }

    pub fn adjust_adam(
        &mut self,
        t: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        utils: WrappedUtils,
    ) {
        safe_lock(&self.layer).adjust_adam(t, learning_rate, beta1, beta2, epsilon, utils);
    }

    /// Saves the layer weights to a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer weights could not be saved to the specified path.
    pub fn save_weights(
        &self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        safe_lock(&self.layer).save_weight(path)
    }

    /// Reads the layer weights from a file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer weights could not be read from the specified path.
    pub fn read_weight(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        safe_lock(&self.layer).read_weight(path)
    }

    pub fn assign_weights(
        &mut self,
        other: &WrappedLayer<NumberEntry, NumberEntry>,
    ) {
        safe_lock(&self.layer).assign_weights(other.clone());
    }

    pub fn assign_trainable_weights(
        &mut self,
        other: &WrappedTrainableLayer<WeightEntry, BiasEntry>,
    ) {
        safe_lock(&self.layer).assign_trainable_weights(other.clone());
    }
}
