use super::activate::ActivationTrait;

/// Sigmoid activation function.
#[derive(Clone)]
pub struct Sigmoid;

impl ActivationTrait for Sigmoid {
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    fn backward(&self, grad_output: &[f64]) -> Vec<f64> {
        grad_output
            .iter()
            .zip(self.forward(grad_output).iter())
            .map(|(&grad, &output)| grad * output * (1.0 - output))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        let input = vec![0.0];
        let output = sigmoid.forward(&input);
        assert!((output[0] - 0.5).abs() < 1e-7);

        let grad_output = vec![1.0];
        let grad_input = sigmoid.backward(&grad_output);
        assert!((grad_input[0] - 0.25).abs() < 1e-7);
    }
}