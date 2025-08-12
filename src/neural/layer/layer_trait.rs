use crate::alloc::allocatable::{Allocatable, WrappedAllocatableTrait};
use crate::neural::mat::matrix::WrappedMatrix;
use crate::neural::utilities::safer::safe_lock;
use crate::neural::utilities::util::WrappedUtils;

use dyn_clone::DynClone;
use std::error::Error;
use std::sync::{Arc, Mutex};
// A trait representing a layer in a neural network.
/// Provides methods for the forward pass, backward pass, weight updates, and layer size information.
pub trait Layer: std::fmt::Debug + DynClone + Allocatable {
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
    fn get_weights(&self) -> WrappedMatrix<f64>;

    /// Returns the biases of the layer.
    fn get_biases(&self) -> Vec<f64>;

    /// Marks the layer as in use.
    fn cleanup(&self);
}

dyn_clone::clone_trait_object!(Layer);

pub trait AllocatableLayer: Allocatable + Layer {
    /// Duplicates the layer, creating a new instance with the same parameters.
    fn duplicate(
        &mut self,
        model_directory: String,
        position_in_nn: usize,
    ) -> Box<dyn AllocatableLayer + Send>;

    /// Copy the layer on the filesystem
    fn copy_on_filesystem(
        &self,
        layer_path: String,
    );
}

#[derive(Debug, Clone)]
pub struct WrappedLayer {
    layer: Arc<Mutex<Box<dyn AllocatableLayer + Send>>>,
}

impl WrappedLayer {
    #[must_use]
    pub fn new(layer: Box<dyn AllocatableLayer + Send>) -> Self {
        Self { layer: Arc::new(Mutex::new(layer)) }
    }

    #[must_use]
    pub fn duplicate(
        &self,
        model_directory: String,
        position_in_nn: usize,
    ) -> Self {
        let new_layer = safe_lock(&self.layer).duplicate(model_directory, position_in_nn);
        Self { layer: Arc::new(Mutex::new(new_layer)) }
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
    pub fn get_weights(&self) -> WrappedMatrix<f64> {
        safe_lock(&self.layer).get_weights()
    }

    #[must_use]
    pub fn get_biases(&self) -> Vec<f64> {
        safe_lock(&self.layer).get_biases()
    }
}

impl WrappedAllocatableTrait for WrappedLayer {
    fn allocate(&self) {
        self.layer.lock().unwrap().allocate();
    }

    fn deallocate(&self) {
        self.layer.lock().unwrap().deallocate();
    }

    fn is_allocated(&self) -> bool {
        self.layer.lock().unwrap().is_allocated()
    }

    fn get_size(&self) -> usize {
        self.layer.lock().unwrap().get_size()
    }

    fn mark_for_use(&mut self) {
        self.layer.lock().unwrap().mark_for_use();
    }

    fn free_from_use(&mut self) {
        self.layer.lock().unwrap().free_from_use();
    }

    fn is_in_use(&self) -> bool {
        self.layer.lock().unwrap().is_in_use()
    }
}

pub trait TrainableLayer: Layer {
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

    /// Assigns the weight of the input other layer
    fn assign_weights(
        &mut self,
        other: WrappedTrainableLayer,
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

dyn_clone::clone_trait_object!(TrainableLayer);

pub trait TrainableAllocatableLayer: Allocatable + TrainableLayer {
    /// Duplicates the layer, creating a new instance with the same parameters.
    fn duplicate(
        &mut self,
        model_directory: String,
        position_in_nn: usize,
    ) -> Box<dyn TrainableAllocatableLayer + Send>;

    /// Copy the layer on the filesystem
    fn copy_on_filesystem(
        &self,
        layer_path: String,
    );
}

#[derive(Debug, Clone)]
pub struct WrappedTrainableLayer {
    layer: Arc<Mutex<Box<dyn TrainableAllocatableLayer + Send>>>,
}

impl WrappedTrainableLayer {
    #[must_use]
    pub fn new(layer: Box<dyn TrainableAllocatableLayer + Send>) -> Self {
        Self { layer: Arc::new(Mutex::new(layer)) }
    }

    #[must_use]
    pub fn duplicate(
        &self,
        model_directory: String,
        position_in_nn: usize,
    ) -> Self {
        let new_layer = safe_lock(&self.layer).duplicate(model_directory, position_in_nn);
        Self { layer: Arc::new(Mutex::new(new_layer)) }
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
    pub fn get_weights(&self) -> WrappedMatrix<f64> {
        safe_lock(&self.layer).get_weights()
    }

    #[must_use]
    pub fn get_biases(&self) -> Vec<f64> {
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

    pub fn assign_weights(
        &mut self,
        other: Self,
    ) {
        safe_lock(&self.layer).assign_weights(other);
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
}

impl WrappedAllocatableTrait for WrappedTrainableLayer {
    fn allocate(&self) {
        safe_lock(&self.layer).allocate();
    }

    fn deallocate(&self) {
        safe_lock(&self.layer).deallocate();
    }

    fn is_allocated(&self) -> bool {
        safe_lock(&self.layer).is_allocated()
    }

    fn get_size(&self) -> usize {
        safe_lock(&self.layer).get_size()
    }

    fn mark_for_use(&mut self) {
        safe_lock(&self.layer).mark_for_use();
    }

    fn free_from_use(&mut self) {
        safe_lock(&self.layer).free_from_use();
    }

    fn is_in_use(&self) -> bool {
        safe_lock(&self.layer).is_in_use()
    }
}
