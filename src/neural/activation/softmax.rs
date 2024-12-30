use crate::neural::nn::shape::ActivationData;

use super::activate::ActivationTrait;

/// Softmax activation function.
#[derive(Debug, Clone)]
pub struct Softmax {
    pub temperature: f64,
    last_output: Option<Vec<f64>>,
}

impl Softmax {
    /// Creates a new Softmax instance with the specified temperature.
    pub fn new(temperature: f64) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive.");
        Self {
            temperature,
            last_output: None,
        }
    }

    /// Applies the softmax function to a vector of inputs.
    fn softmax(&self, input: &[f64]) -> Vec<f64> {
        let max_input = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = input
            .iter()
            .map(|&x| ((x - max_input) / self.temperature).exp())
            .collect();
        let sum_exp = exp_values.iter().sum::<f64>();
        exp_values.into_iter().map(|v| v / sum_exp).collect()
    }
}

impl ActivationTrait for Softmax {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output = self.softmax(input);
        self.last_output = Some(output.clone()); // Cache the output
        output
    }

    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let softmax_output = self
            .last_output
            .as_ref()
            .expect("Forward must be called before backward to cache the output.");

        let len = softmax_output.len();
        let mut grad_input = vec![0.0; len];

        for i in 0..len {
            for j in 0..len {
                if i == j {
                    grad_input[i] += grad_output[j] * softmax_output[i] * (1.0 - softmax_output[i]);
                } else {
                    grad_input[i] += -grad_output[j] * softmax_output[i] * softmax_output[j];
                }
            }
        }

        self.last_output = None; // Clear the cache

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

        println!("Softmax output: {:?}", output);

        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-7, "Softmax outputs should sum to 1.");
    }

    #[test]
    fn test_softmax_backward() {
        let mut softmax = Softmax::new(1.0);
        let input = vec![1.0, 2.0, 3.0];
        softmax.forward(input.as_slice());
        let grad_output = vec![0.1, 0.2, 0.7];
        let grad_input = softmax.backward(&grad_output);

        println!("Softmax backward output: {:?}", grad_input);
        assert_eq!(
            grad_input.len(),
            input.len(),
            "Gradient input length should match input length."
        );
    }
}
