/// Tanh activation function.
pub struct Tanh;

impl Activation for Tanh {
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| x.tanh()).collect()
    }

    fn backward(&self, grad_output: &[f64]) -> Vec<f64> {
        grad_output
            .iter()
            .zip(self.forward(grad_output).iter())
            .map(|(&grad, &output)| grad * (1.0 - output.powi(2)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh() {
        let tanh = Tanh;
        let input = vec![0.0];
        let output = tanh.forward(&input);
        assert!((output[0] - 0.0).abs() < 1e-7);

        let grad_output = vec![1.0];
        let grad_input = tanh.backward(&grad_output);
        assert!((grad_input[0] - 1.0).abs() < 1e-7);
    }
}