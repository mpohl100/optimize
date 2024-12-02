use crate::neural::nn::shape::ActivationType;

use dyn_clone::DynClone;

/// A trait for activation functions used in neural networks.
/// Provides methods for forward pass (activation) and backward pass (gradient computation).
pub trait ActivationTrait: std::fmt::Debug + DynClone {
    /// Applies the activation function to the input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to a vector of `f64` values representing the input.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values after applying the activation function element-wise.
    fn forward(&self, input: &[f64]) -> Vec<f64>;

    /// Computes the gradient of the activation function for backpropagation.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - A reference to a vector of `f64` values representing the gradient of the loss
    ///   with respect to the output of this activation function.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values representing the gradient of the loss with respect to the input.
    fn backward(&self, grad_output: &[f64]) -> Vec<f64>;

    /// Returns the name of the activation function.
    fn get_activation_type(&self) -> ActivationType;
}

dyn_clone::clone_trait_object!(ActivationTrait);
