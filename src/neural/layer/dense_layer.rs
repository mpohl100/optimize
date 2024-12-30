use super::layer_trait::Layer;
pub use crate::neural::mat::matrix::Matrix;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

/// A fully connected neural network layer (Dense layer).
#[derive(Debug, Clone)]
pub struct DenseLayer {
    weights: Matrix<f64>,             // Weight matrix (output_size x input_size)
    biases: Vec<f64>,                 // Bias vector (output_size)
    input_cache: Vec<f64>,            // Cache input for use in backward pass
    input_batch_cache: Vec<Vec<f64>>, // Cache batch input for use in backward pass
    weight_grads: Matrix<f64>,        // Gradient of weights
    bias_grads: Vec<f64>,             // Gradient of biases
}

impl DenseLayer {
    /// Creates a new DenseLayer with given input and output sizes.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Create a dense layer with default weights
        let mut dense_layer = DenseLayer {
            weights: Matrix::new(output_size, input_size),
            biases: vec![0.0; output_size],
            input_cache: vec![],
            input_batch_cache: vec![],
            weight_grads: Matrix::new(output_size, input_size),
            bias_grads: vec![0.0; output_size],
        };

        // Initialize weights with random values in [-0.5, 0.5]
        dense_layer.initialize_weights();
        dense_layer
    }

    /// Initialize the weights with random values in the range [-0.5, 0.5]
    fn initialize_weights(&mut self) {
        let mut rng = rand::thread_rng();
        // initialize weights from -0.5 to 0.5
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                *self.weights.get_mut_unchecked(i, j) = rng.gen_range(-0.5..0.5);
            }
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.input_cache = input.to_vec(); // Cache the input for backpropagation
        self.weights
            .iter()
            .enumerate() // Include the row index in the iteration
            .map(|(row_idx, weights_row)| {
                weights_row
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + self.biases[row_idx] // Use the bias corresponding to the row index
            })
            .collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_batch(&mut self, input: &[f64]) -> Vec<f64> {
        // Store input for potential use in backward pass (not needed in this function)
        self.input_batch_cache.push(input.to_vec().clone());

        // Initialize the output vector with the size of biases
        let mut output = vec![0.0; self.biases.len()];

        let num_rows = self.weights.rows();
        let num_cols = self.weights.cols();
        // Check dimensions or panic:
        assert_eq!(num_rows, self.biases.len());
        assert_eq!(num_cols, input.len());

        // Iterate over each element in biases
        for i in 0..num_rows {
            // Initialize output[i] with the corresponding bias value
            output[i] = self.biases[i];

            // Accumulate the dot product of weights and input
            for j in 0..num_cols {
                output[i] += self.weights.get_unchecked(i, j) * input[j];
            }
        }

        output
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

    #[allow(clippy::needless_range_loop)]
    fn backward_batch(&mut self, grad_output: &[f64]) -> Vec<f64> {
        // Initialize grad_input with the size of input_cache, filled with zeros
        let mut grad_input = vec![0.0; self.input_cache.len()];

        let num_rows = self.weights.rows();
        let num_cols = self.weights.cols();
        let last_input_cache = &self.input_batch_cache[self.input_batch_cache.len() - 1];
        // Check dimensions or panic:
        assert_eq!(num_rows, self.biases.len());
        assert_eq!(num_cols, last_input_cache.len());

        // Calculate gradients for weights and biases
        for i in 0..num_rows {
            for j in 0..num_cols {
                // Update weight gradients
                *self.weight_grads.get_mut_unchecked(i, j) += grad_output[i] * last_input_cache[j];
                grad_input[j] += self.weights.get_unchecked(i, j) * grad_output[i];
            }
            // Update bias gradients
            self.bias_grads[i] += grad_output[i];
        }

        grad_input
    }

    #[allow(clippy::needless_range_loop)]
    fn update_weights(&mut self, learning_rate: f64) {
        for weights_row in self.weights.iter_mut() {
            for weight in weights_row.iter_mut() {
                *weight -= learning_rate * *weight;
            }
        }
        for bias in self.biases.iter_mut() {
            *bias -= learning_rate * *bias;
        }
        self.input_cache.clear();
        self.input_batch_cache.clear();
    }

    fn input_size(&self) -> usize {
        self.input_cache.len()
    }

    fn output_size(&self) -> usize {
        self.weights.rows()
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        // Save weights and biases to a file at the specified path
        let mut file = File::create(path)?;
        writeln!(file, "{} {}", self.weights.rows(), self.weights.cols())?;
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                write!(file, "{} ", self.weights.get_unchecked(i, j))?;
            }
            writeln!(file)?;
        }
        for i in 0..self.biases.len() {
            write!(file, "{} ", self.biases[i])?;
        }
        writeln!(file)?;
        Ok(())
    }

    fn read(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        if let Some(Ok(line)) = lines.next() {
            let mut parts = line.split_whitespace();
            let rows = parts.next().unwrap().parse::<usize>()?;
            let cols = parts.next().unwrap().parse::<usize>()?;
            self.weights = Matrix::new(rows, cols);
            for i in 0..rows {
                if let Some(Ok(line)) = lines.next() {
                    let mut parts = line.split_whitespace();
                    for j in 0..cols {
                        if let Some(part) = parts.next() {
                            *self.weights.get_mut_unchecked(i, j) = part.parse::<f64>()?;
                        }
                    }
                }
            }
        }
        if let Some(Ok(line)) = lines.next() {
            let mut parts = line.split_whitespace();
            for i in 0..self.biases.len() {
                if let Some(part) = parts.next() {
                    self.biases[i] = part.parse::<f64>()?;
                }
            }
        }
        Ok(())
    }

    fn resize(&mut self, input_size: usize, output_size: usize) {
        let old_weights = self.weights.clone();
        let old_biases = self.biases.clone();
        // Resize the layer to the input dimensions
        self.weights = Matrix::new(output_size, input_size);
        self.biases = vec![0.0; output_size];
        self.input_cache = vec![0.0; input_size];
        self.weight_grads = Matrix::new(output_size, input_size);
        self.bias_grads = vec![0.0; output_size];

        for i in 0..output_size {
            for j in 0..input_size {
                if i < old_weights.rows() && j < old_weights.cols() {
                    *self.weights.get_mut_unchecked(i, j) = *old_weights.get_unchecked(i, j);
                }
            }
            if i < old_biases.len() {
                self.biases[i] = old_biases[i];
            }
        }
    }

    fn assign_weights(&mut self, other: &dyn Layer) {
        let weights = other.get_weights();

        let biases = other.get_biases();

        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    *self.weights.get_mut_unchecked(i, j) = *weights.get_unchecked(i, j);
                }
            }
            if i < biases.len() {
                self.biases[i] = biases[i];
            }
        }
    }

    fn get_weights(&self) -> Matrix<f64> {
        self.weights.clone()
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.clone()
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
