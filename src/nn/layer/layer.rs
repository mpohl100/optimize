/// A trait representing a layer in a neural network.
/// Provides methods for the forward pass, backward pass, weight updates, and layer size information.
pub trait Layer where Self: Clone {
    /// Performs the forward pass of the layer, computing the output based on the input vector.
    ///
    /// # Arguments
    ///
    /// * `input` - A reference to a vector of `f64` values representing the input data.
    ///
    /// # Returns
    ///
    /// * A vector of `f64` values representing the output of the layer.
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

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
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64>;

    /// Updates the weights of the layer based on the specified learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - A `f64` value representing the learning rate for weight updates.
    fn update_weights(&mut self, learning_rate: f64);

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
}
