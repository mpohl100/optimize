use rand::Rng;

/// A fully connected neural network layer (Dense layer).
pub struct DenseLayer {
    weights: Vec<Vec<f64>>, // Weight matrix (output_size x input_size)
    biases: Vec<f64>,       // Bias vector (output_size)
    input_cache: Vec<f64>,  // Cache input for use in backward pass
}

impl DenseLayer {
    /// Creates a new DenseLayer with given input and output sizes.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();
        let biases = vec![0.0; output_size];
        let input_cache = vec![0.0; input_size];

        Self {
            weights,
            biases,
            input_cache,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.input_cache = input.to_vec(); // Cache the input for backpropagation

        self.weights
            .iter()
            .map(|weights_row| {
                weights_row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum::<f64>()
                    + self.biases[weights_row.len() - 1]
            })
            .collect()
    }

    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        // Compute the gradient with respect to the input
        let grad_input = (0..self.input_cache.len())
            .map(|i| {
                self.weights
                    .iter()
                    .zip(grad_output.iter())
                    .map(|(weights_row, &grad)| weights_row[i] * grad)
                    .sum()
            })
            .collect();

        // Compute the gradient with respect to the weights and biases
        for (grad, weights_row) in grad_output.iter().zip(self.weights.iter_mut()) {
            for (i, weight) in weights_row.iter_mut().enumerate() {
                *weight += grad * self.input_cache[i];
            }
        }

        grad_input
    }

    fn update_weights(&mut self, learning_rate: f64) {
        for weights_row in self.weights.iter_mut() {
            for weight in weights_row.iter_mut() {
                *weight -= learning_rate * *weight;
            }
        }
        for bias in self.biases.iter_mut() {
            *bias -= learning_rate * *bias;
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
