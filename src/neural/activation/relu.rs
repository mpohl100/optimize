use crate::neural::nn::shape::ActivationData;
use crate::neural::nn::shape::ActivationType;

use super::activate::ActivationTrait;

/// `ReLU` (Rectified Linear Unit) activation function.
#[derive(Debug, Clone)]
pub struct ReLU {
    // Cache for the input during forward pass, used during backward pass
    input_cache: Option<Vec<f64>>,
}

impl ReLU {
    #[must_use] pub const fn new() -> Self {
        Self { input_cache: None }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationTrait for ReLU {
    fn forward(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        // Cache the input for use in backpropagation
        self.input_cache = Some(input.to_vec());

        // Apply ReLU function: output x if x > 0, else 0
        input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
    }

    fn backward(
        &mut self,
        grad_output: &[f64],
    ) -> Vec<f64> {
        // Retrieve the cached input from the forward pass
        let input = match &self.input_cache {
            Some(input) => input,
            None => panic!("ReLU forward must be called before backward."),
        };

        // Compute the gradient for each element: pass the gradient if input > 0, else 0
        grad_output
            .iter()
            .zip(input.iter())
            .map(|(&_grad, &inp)| if inp > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }

    fn get_activation_data(&self) -> ActivationData {
        ActivationData::new(ActivationType::ReLU)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let mut relu = ReLU::new();
        let input = vec![-1.0, 0.0, 1.0];
        let output = relu.forward(&input);
        // print output
        println!("{:?}", output);
        assert_eq!(output, vec![0.0, 0.0, 1.0]);

        let grad_output = vec![-0.5, 0.0, 0.5];
        let grad_input = relu.backward(&grad_output);
        // print grad_input
        println!("{:?}", grad_input);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0]);
    }
}
