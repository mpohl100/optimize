use super::layer_trait::Layer;
use rand::Rng;

/// A fully connected neural network layer (Dense layer).
#[derive(Clone)]
pub struct DenseLayer {
    weights: Vec<Vec<f64>>,      // Weight matrix (output_size x input_size)
    biases: Vec<f64>,            // Bias vector (output_size)
    input_cache: Vec<f64>,       // Cache input for use in backward pass
    weight_grads: Vec<Vec<f64>>, // Gradient of weights
    bias_grads: Vec<f64>,        // Gradient of biases
}

impl DenseLayer {
    /// Creates a new DenseLayer with given input and output sizes.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        // initalize weights as a Vec<Vec<f64>> with random values between -0.1 and 0.1
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();
        let biases = vec![0.0; output_size];
        let input_cache = vec![0.0; input_size];
        let weight_grads = vec![vec![0.0; input_size]; output_size];
        let bias_grads = vec![0.0; output_size];

        // check dimensions or panic:
        assert_eq!(weights.len(), output_size);
        for weights_row in weights.iter() {
            assert_eq!(weights_row.len(), input_size);
        }
        assert_eq!(biases.len(), output_size);
        assert!(input_cache.len() == input_size);

        assert!(weight_grads.len() == output_size);
        for weight_grads_row in weight_grads.iter() {
            assert!(weight_grads_row.len() == input_size);
        }
        assert!(bias_grads.len() == output_size);

        Self {
            weights,
            biases,
            input_cache,
            weight_grads,
            bias_grads,
        }
    }
}

impl Layer for DenseLayer {
    #[allow(clippy::needless_range_loop)]
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        // Store input for potential use in backward pass (not needed in this function)
        self.input_cache = input.to_vec().clone();

        // Initialize the output vector with the size of biases
        let mut output = vec![0.0; self.biases.len()];

        // Iterate over each element in biases
        for i in 0..self.biases.len() {
            // Initialize output[i] with the corresponding bias value
            output[i] = self.biases[i];

            // Accumulate the dot product of weights and input
            for j in 0..input.len() {
                output[i] += self.weights[i][j] * input[j];
            }
        }

        output
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        // Initialize grad_input with the size of input_cache, filled with zeros
        let mut grad_input = vec![0.0; self.input_cache.len()];

        // Calculate gradients for weights and biases
        for i in 0..self.weights.len() {
            for j in 0..self.input_cache.len() {
                // Update weight gradients
                self.weight_grads[i][j] += grad_output[i] * self.input_cache[j];
            }
            // Update bias gradients
            self.bias_grads[i] += grad_output[i];
        }

        // Calculate gradient with respect to the input for backpropagation
        for i in 0..self.weights.len() {
            for j in 0..self.input_cache.len() {
                grad_input[j] += self.weights[i][j] * grad_output[i];
            }
        }

        grad_input
    }

    #[allow(clippy::needless_range_loop)]
    fn update_weights(&mut self, learning_rate: f64) {
        // Update weights and biases using the accumulated gradients
        for i in 0..self.weights.len() {
            for j in 0..self.weights[0].len() {
                // Update weight with gradient descent step
                self.weights[i][j] -= learning_rate * self.weight_grads[i][j];
                // Reset weight gradient after update
                self.weight_grads[i][j] = 0.0;
            }
            // Update biases and reset bias gradients
            self.biases[i] -= learning_rate * self.bias_grads[i];
            self.bias_grads[i] = 0.0;
        }
    }

    fn input_size(&self) -> usize {
        self.input_cache.len()
    }

    fn output_size(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer() {
        let mut layer = DenseLayer::new(3, 2);

        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);

        let grad_output = vec![0.1, 0.2];
        let grad_input = layer.backward(&grad_output);

        assert_eq!(grad_input.len(), 3);

        layer.update_weights(0.01);
    }
}
