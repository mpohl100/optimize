use super::activate::ActivationTrait;
use crate::neural::nn::shape::ActivationData;

/// Softmax activation function.
#[derive(Debug, Clone)]
pub struct Softmax {
    pub temperature: f64,
    cached_output: Option<Vec<f64>>, // Cache the output of the forward pass
}

impl Softmax {
    /// Creates a new Softmax instance with the specified temperature.
    ///
    /// # Panics
    ///
    /// Panics if `temperature` is not positive.
    #[must_use]
    pub fn new(temperature: f64) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive.");
        Self { temperature, cached_output: None }
    }

    /// Applies the softmax function to a vector of inputs.
    fn softmax(
        &self,
        input: &[f64],
    ) -> Vec<f64> {
        let max_input = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> =
            input.iter().map(|&x| ((x - max_input) / self.temperature).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f64>();
        exp_values.into_iter().map(|v| v / sum_exp).collect()
    }
}

impl ActivationTrait for Softmax {
    fn forward(
        &mut self,
        input: &[f64],
    ) -> Vec<f64> {
        let output = self.softmax(input);
        self.cached_output = Some(output.clone()); // Cache the output for the backward pass
        output
    }

    fn backward(
        &mut self,
        grad_output: &[f64],
    ) -> Vec<f64> {
        let softmax_output = match &self.cached_output {
            Some(output) => output.clone(),
            None => panic!("Softmax forward must be called before backward."),
        };

        let mut grad_input = vec![0.0; grad_output.len()];
        for i in 0..softmax_output.len() {
            for j in 0..softmax_output.len() {
                if i == j {
                    grad_input[i] += softmax_output[i] * (1.0 - softmax_output[i]) * grad_output[j];
                } else {
                    grad_input[i] += -softmax_output[i] * softmax_output[j] * grad_output[j];
                }
            }
            // Adjust gradients for temperature
            grad_input[i] /= self.temperature;
        }
        grad_input
    }

    fn get_activation_data(&self) -> ActivationData {
        ActivationData::new_softmax(self.temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let mut softmax = Softmax::new(1.0);
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax.forward(&input);

        println!("Softmax output: {output:?}");

        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-7, "Softmax outputs should sum to 1.");
        assert!(output.iter().all(|&v| v > 0.0), "Softmax probabilities must be positive.");
    }

    #[test]
    fn test_softmax_with_temperature() {
        let mut softmax = Softmax::new(0.5);
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax.forward(&input);

        println!("Softmax output with temperature 0.5: {output:?}");

        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-7, "Softmax outputs should sum to 1.");
        assert!(output.iter().all(|&v| v > 0.0), "Softmax probabilities must be positive.");
    }

    #[test]
    fn test_softmax_backward() {
        let mut softmax = Softmax::new(1.0);
        let input = vec![1.0, 2.0, 3.0];
        softmax.forward(input.as_slice());
        let grad_output = vec![0.1, 0.2, 0.7];
        let grad_input = softmax.backward(&grad_output);

        println!("Softmax backward output: {grad_input:?}");
        assert_eq!(
            grad_input.len(),
            input.len(),
            "Gradient input length should match input length."
        );
    }
}
