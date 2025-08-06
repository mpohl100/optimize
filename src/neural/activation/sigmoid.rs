use crate::neural::nn::shape::ActivationData;
use crate::neural::nn::shape::ActivationType;

use super::activate::ActivationTrait;

/// Sigmoid activation function.
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    /// Creates a new Sigmoid instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn sigmoid_vec(
        &self,
        input: &[f64],
    ) -> Vec<f64> {
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivationTrait for Sigmoid {
    fn forward(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        self.sigmoid_vec(input)
    }

    fn backward(
        &mut self,
        grad_output: &[f64],
    ) -> Vec<f64> {
        grad_output
            .iter()
            .zip(self.sigmoid_vec(grad_output).iter())
            .map(|(&grad, &output)| grad * output * (1.0 - output))
            .collect()
    }

    fn get_activation_data(&self) -> ActivationData {
        ActivationData::new(ActivationType::Sigmoid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mut sigmoid = Sigmoid;
        let input = vec![0.0];
        let output = sigmoid.forward(&input);
        // print output
        println!("{:?}", output);
        assert!((output[0] - 0.5).abs() < 1e-7);

        let grad_output = vec![1.0];
        let grad_input = sigmoid.backward(&grad_output);
        // print grad_input
        println!("{:?}", grad_input);
        assert!((grad_input[0] - 0.196612).abs() < 1e-7);
    }
}
