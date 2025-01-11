use super::layer_trait::Layer;
use super::layer_trait::TrainableLayer;
pub use crate::neural::mat::matrix::Matrix;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct DenseLayer {
    weights: Matrix<f64>,
    biases: Vec<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        DenseLayer {
            weights: Matrix::new(output_size, input_size),
            biases: vec![0.0; output_size],
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
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

    fn forward_batch(&mut self, _input: &[f64]) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.weights.cols()
    }

    fn output_size(&self) -> usize {
        self.weights.rows()
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        save(path, &self.weights, &self.biases)
    }

    fn read(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read(path)?;
        self.weights = weights;
        self.biases = biases;
        Ok(())
    }

    fn get_weights(&self) -> Matrix<f64> {
        self.weights.clone()
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.clone()
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct Weight{
    value: f64,
    grad: f64,
    m: f64,
    v: f64,
}

#[derive(Default, Debug, Clone, Copy)]
struct Bias{
    value: f64,
    grad: f64,
    m: f64,
    v: f64,
}

/// A fully connected neural network layer (Dense layer).
#[derive(Debug, Clone)]
pub struct TrainableDenseLayer {
    weights: Matrix<Weight>,             // Weight matrix (output_size x input_size)
    biases: Vec<Bias>,                 // Bias vector (output_size)
    input_cache: Vec<f64>,            // Cache input for use in backward pass
    input_batch_cache: Vec<Vec<f64>>, // Cache batch input for use in backward pass
}

impl TrainableDenseLayer {
    /// Creates a new TrainableDenseLayer with given input and output sizes.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Create a dense layer with default weights
        let mut dense_layer = TrainableDenseLayer {
            weights: Matrix::new(output_size, input_size),
            biases: vec![Bias::default(); output_size],
            input_cache: vec![],
            input_batch_cache: vec![],
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
                self.weights.get_mut_unchecked(i, j).value = rng.gen_range(-0.5..0.5);
            }
        }
    }
}

impl Layer for TrainableDenseLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.input_cache = input.to_vec(); // Cache the input for backpropagation
        self.weights
            .iter()
            .enumerate() // Include the row index in the iteration
            .map(|(row_idx, weights_row)| {
                weights_row
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w.value * x)
                    .sum::<f64>()
                    + self.biases[row_idx].value // Use the bias corresponding to the row index
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
            output[i] = self.biases[i].value;

            // Accumulate the dot product of weights and input
            for j in 0..num_cols {
                output[i] += self.weights.get_unchecked(i, j).value * input[j];
            }
        }

        output
    }

    fn input_size(&self) -> usize {
        self.input_cache.len()
    }

    fn output_size(&self) -> usize {
        self.weights.rows()
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        // assign weights and biases to a matrix and vector
        let mut weights = Matrix::new(self.weights.rows(), self.weights.cols());
        let mut biases = vec![0.0; self.biases.len()];
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                *weights.get_mut_unchecked(i, j) = self.weights.get_unchecked(i, j).value;
            }
            biases[i] = self.biases[i].value;
        }
        save(path, &weights, &biases)
    }

    fn read(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read(path)?;
        // assign all weights and biases
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    self.weights.get_mut_unchecked(i, j).value = *weights.get_unchecked(i, j);
                }
            }
            if i < biases.len() {
                self.biases[i].value = biases[i];
            }
        }
        Ok(())
    }

    fn get_weights(&self) -> Matrix<f64> {
        let mut weights = Matrix::new(self.weights.rows(), self.weights.cols());
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                *weights.get_mut_unchecked(i, j) = self.weights.get_unchecked(i, j).value;
            }
        }
        weights
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.iter().map(|bias| bias.value).collect()
    }
}

impl TrainableLayer for TrainableDenseLayer {
    /// Backward pass for the dense layer
    ///
    /// - `d_out`: Gradient of the loss with respect to the output of this layer
    /// - Returns: Gradient of the loss with respect to the input of this layer
    fn backward(&mut self, d_out: &[f64]) -> Vec<f64> {
        // Calculate weight gradients
        for (i, row_grad) in self.weights.iter_mut().enumerate() {
            for (j, grad) in row_grad.iter_mut().enumerate() {
                grad.grad = d_out[i] * self.input_cache[j];
            }
        }

        // Calculate input gradients
        let mut d_input = vec![0.0; self.input_cache.len()];
        for (i, weights_row) in self.weights.iter().enumerate() {
            for (j, &weight) in weights_row.iter().enumerate() {
                d_input[j] += weight.value * d_out[i];
            }
        }

        d_input
    }

    /// Update weights and biases using their respective gradients
    ///
    /// - `learning_rate`: The step size for gradient descent
    fn update_weights(&mut self, learning_rate: f64) {
        // Update weight
        for weights_row in self.weights.iter_mut() {
            for weight in weights_row.iter_mut() {
                weight.value -= learning_rate * weight.grad;
            }
        }

        // Update biases
        for bias in self.biases.iter_mut() {
            bias.value -= learning_rate * bias.grad;
        }
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
                self.weights.get_mut_unchecked(i, j).grad += grad_output[i] * last_input_cache[j];
                grad_input[j] += self.weights.get_unchecked(i, j).value * grad_output[i];
            }
            // Update bias gradients
            self.biases[i].grad += grad_output[i];
        }

        grad_input
    }

    fn resize(&mut self, input_size: usize, output_size: usize) {
        let old_weights = self.weights.clone();
        let old_biases = self.biases.clone();
        // Resize the layer to the input dimensions
        self.weights = Matrix::new(output_size, input_size);
        self.biases = vec![Bias::default(); output_size];
        self.input_cache = vec![0.0; input_size];

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

    fn assign_weights(&mut self, other: &dyn TrainableLayer) {
        let weights = other.get_weights();

        let biases = other.get_biases();

        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    self.weights.get_mut_unchecked(i, j).value = *weights.get_unchecked(i, j);
                }
            }
            if i < biases.len() {
                self.biases[i].value = biases[i];
            }
        }
    }

    fn adjust_adam(&mut self, t: usize, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) {
        // Update weights
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                let grad = self.weights.get_unchecked(i, j).grad;

                // Update first and second moments
                self.weights.get_mut_unchecked(i, j).m =
                    beta1 * self.weights.get_unchecked(i, j).m + (1.0 - beta1) * grad;
                self.weights.get_mut_unchecked(i, j).v =
                    beta2 * self.weights.get_unchecked(i, j).v + (1.0 - beta2) * grad.powi(2);

                // Bias correction
                let m_hat = self.weights.get_unchecked(i, j).m / (1.0 - beta1.powi(t as i32));
                let v_hat = self.weights.get_unchecked(i, j).v / (1.0 - beta2.powi(t as i32));

                // Adjusted learning rate
                let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);

                // Update weights
                self.weights.get_mut_unchecked(i, j).value -= adjusted_learning_rate * m_hat;
            }
        }

        // Update biases
        for i in 0..self.biases.len() {
            let grad = self.biases[i].grad;

            // Update first and second moments
            self.biases[i].m = beta1 * self.biases[i].m + (1.0 - beta1) * grad;
            self.biases[i].v = beta2 * self.biases[i].v + (1.0 - beta2) * grad.powi(2);

            // Bias correction
            let m_hat = self.biases[i].m / (1.0 - beta1.powi(t as i32));
            let v_hat = self.biases[i].v / (1.0 - beta2.powi(t as i32));

            // Adjusted learning rate
            let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);

            // Update biases
            self.biases[i].value -= adjusted_learning_rate * m_hat;
        }
    }
}

fn save(path: &str, weights: &Matrix<f64>, biases: &[f64]) -> Result<(), Box<dyn Error>> {
    // Save weights and biases to a file at the specified path
    let mut file = File::create(path)?;
    writeln!(file, "{} {}", weights.rows(), weights.cols())?;
    for i in 0..weights.rows() {
        for j in 0..weights.cols() {
            write!(file, "{} ", weights.get_unchecked(i, j))?;
        }
        writeln!(file)?;
    }
    for bias in biases.iter() {
        write!(file, "{} ", bias)?;
    }
    writeln!(file)?;
    Ok(())
}

fn read(path: &str) -> Result<(Matrix<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut weights = Matrix::new(1, 1);
    let mut biases = vec![0.0; 1];
    if let Some(Ok(line)) = lines.next() {
        let mut parts = line.split_whitespace();
        let rows = parts.next().unwrap().parse::<usize>()?;
        let cols = parts.next().unwrap().parse::<usize>()?;
        weights = Matrix::new(rows, cols);
        for i in 0..rows {
            if let Some(Ok(line)) = lines.next() {
                let mut parts = line.split_whitespace();
                for j in 0..cols {
                    if let Some(part) = parts.next() {
                        *weights.get_mut_unchecked(i, j) = part.parse::<f64>()?;
                    }
                }
            }
        }
    }
    if let Some(Ok(line)) = lines.next() {
        let mut parts = line.split_whitespace();
        biases = vec![0.0; weights.rows()];
        for bias in &mut biases {
            if let Some(part) = parts.next() {
                *bias = part.parse::<f64>()?;
            }
        }
    }
    Ok((weights, biases))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer() {
        let mut layer = TrainableDenseLayer::new(3, 2);

        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);

        let grad_output = vec![0.1, 0.2];
        let grad_input = layer.backward(&grad_output);

        assert_eq!(grad_input.len(), 3);

        layer.update_weights(0.01);
    }
}
