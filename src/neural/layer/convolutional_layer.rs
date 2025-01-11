use super::layer_trait::Layer;
use super::layer_trait::TrainableLayer;
pub use crate::neural::mat::matrix::Matrix;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

#[derive(Default, Debug, Clone, Copy)]
struct ConvWeight {
    value: f64,
    grad: f64,
    m: f64,
    v: f64,
}

#[derive(Default, Debug, Clone, Copy)]
struct ConvBias {
    value: f64,
    grad: f64,
    m: f64,
    v: f64,
}

/// A convolutional neural network layer.
#[derive(Debug, Clone)]
pub struct ConvolutionalLayer {
    kernels: Vec<Matrix<ConvWeight>>, // Kernels (filters) for the convolution
    biases: Vec<ConvBias>,           // Biases for each output channel
    input_cache: Vec<Vec<f64>>,      // Cached input for backpropagation
    kernel_size: usize,              // Size of the kernel (assumed square for simplicity)
    stride: usize,                   // Stride of the convolution
    padding: usize,                  // Padding added to the input
    input_shape: (usize, usize, usize), // Input dimensions: (height, width, channels)
    output_shape: (usize, usize, usize), // Output dimensions: (height, width, channels)
}

impl ConvolutionalLayer {
    /// Creates a new ConvolutionalLayer with specified parameters.
    pub fn new(
        input_shape: (usize, usize, usize), // (height, width, channels)
        output_channels: usize,             // Number of output channels
        kernel_size: usize,                 // Size of each kernel (assumed square)
        stride: usize,                      // Stride for the convolution
        padding: usize                      // Padding for the input
    ) -> Self {
        let (input_height, input_width, input_channels) = input_shape;

        // Compute output dimensions
        let output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1;
        let output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1;

        // Initialize kernels and biases
        let mut rng = rand::thread_rng();
        let mut kernels = vec![];
        for _ in 0..output_channels {
            kernels.push(Matrix::new_filled(
                kernel_size * kernel_size,
                input_channels,
                || ConvWeight {
                    value: rng.gen_range(-0.5..0.5),
                    grad: 0.0,
                    m: 0.0,
                    v: 0.0,
                },
            ));
        }

        let biases = vec![ConvBias::default(); output_channels];

        Self {
            kernels,
            biases,
            input_cache: vec![],
            kernel_size,
            stride,
            padding,
            input_shape,
            output_shape: (output_height, output_width, output_channels),
        }
    }

    /// Perform forward pass for a single input.
    pub fn convolve(
        &self,
        input: &Matrix<f64>,
        kernel: &Matrix<ConvWeight>,
        bias: &ConvBias,
        output_height: usize,
        output_width: usize,
    ) -> Matrix<f64> {
        let mut output = Matrix::new(output_height, output_width);
        for oh in 0..output_height {
            for ow in 0..output_width {
                let mut sum = 0.0;
                for kh in 0..self.kernel_size {
                    for kw in 0..self.kernel_size {
                        for c in 0..self.input_shape.2 {
                            let ih = oh * self.stride + kh - self.padding;
                            let iw = ow * self.stride + kw - self.padding;
                            if ih >= 0 && ih < self.input_shape.0 && iw >= 0 && iw < self.input_shape.1 {
                                sum += input.get(ih, iw, c) * kernel.get(kh * self.kernel_size + kw, c).value;
                            }
                        }
                    }
                }
                output.set(oh, ow, sum + bias.value);
            }
        }
        output
    }

    /// Perform a backward pass for gradients computation.
    pub fn backward_convolve(
        &mut self,
        d_out: &Matrix<f64>,
        input: &Matrix<f64>,
    ) -> Matrix<f64> {
        let kernel_size = (self.kernel.cols() as f64).sqrt() as usize;
        let output_width = ((input_width + 2 * self.padding - kernel_size) / self.stride) + 1;
        let output_height = ((input_height + 2 * self.padding - kernel_size) / self.stride) + 1;

        // Initialize gradients for input
        let mut d_input = vec![vec![0.0; input_width * input_height]; input.len()];

        for (batch_idx, d_output_batch) in d_output.iter().enumerate() {
            for oc in 0..self.biases.len() {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let d_out_val = d_output_batch[oc * output_width * output_height + y * output_width + x];

                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let in_y = y * self.stride + ky - self.padding;
                                let in_x = x * self.stride + kx - self.padding;

                                if in_y >= 0 && in_x >= 0 && in_y < input_height as isize && in_x < input_width as isize {
                                    let in_y = in_y as usize;
                                    let in_x = in_x as usize;

                                    for ic in 0..self.kernel.rows() / kernel_size {
                                        let kernel_val = self.kernel.get_unchecked(oc, ic * kernel_size * kernel_size + ky * kernel_size + kx);

                                        d_input[batch_idx][ic * input_width * input_height + in_y * input_width + in_x] += kernel_val * d_out_val;

                                        // Update kernel gradient
                                        let input_val = input[batch_idx][ic * input_width * input_height + in_y * input_width + in_x];
                                        let kernel_grad = d_out_val * input_val;
                                        self.kernel.get_mut_unchecked(oc, ic * kernel_size * kernel_size + ky * kernel_size + kx).grad += kernel_grad;
                                    }
                                }
                            }
                        }

                        // Update bias gradient
                        self.biases[oc] += d_out_val;
                    }
                }
            }
        }

        d_input
    }

    /// Update weights and biases using gradients.
    pub fn update_weights(&mut self, learning_rate: f64) {
        for kernel in &mut self.kernels {
            for weight in kernel.iter_mut() {
                weight.value -= learning_rate * weight.grad;
            }
        }
        for bias in &mut self.biases {
            bias.value -= learning_rate * bias.grad;
        }
    }
}

impl Layer for ConvolutionalLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let input_matrix = Matrix::from_vec(
            self.input_shape.0,
            self.input_shape.1,
            self.input_shape.2,
            input,
        );

        // Cache input for backward pass
        self.input_cache.push(input.to_vec());

        let mut output = vec![];
        for (kernel, bias) in self.kernels.iter().zip(self.biases.iter()) {
            let conv_output = self.convolve(
                &input_matrix,
                kernel,
                bias,
                self.output_shape.0,
                self.output_shape.1,
            );
            output.extend_from_slice(&conv_output.to_vec());
        }

        output
    }

    fn forward_batch(&mut self, _input: &[f64]) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.input_shape.0 * self.input_shape.1 * self.input_shape.2
    }

    fn output_size(&self) -> usize {
        self.output_shape.0 * self.output_shape.1 * self.output_shape.2
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        // TODO: Implement saving logic for kernels and biases
        unimplemented!()
    }

    fn read(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        // TODO: Implement reading logic for kernels and biases
        unimplemented!()
    }

    fn get_weights(&self) -> Matrix<f64> {
        unimplemented!()
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.iter().map(|b| b.value).collect()
    }
}

impl TrainableLayer for ConvolutionalLayer {
    fn backward(&mut self, d_out: &[f64]) -> Vec<f64> {
        let d_out_matrix = Matrix::from_vec(
            self.output_shape.0,
            self.output_shape.1,
            self.output_shape.2,
            d_out,
        );
        let input_matrix = Matrix::from_vec(
            self.input_shape.0,
            self.input_shape.1,
            self.input_shape.2,
            &self.input_cache.last().unwrap(),
        );
        
        let d_input = self.backward_convolve(&d_out_matrix, &input_matrix);
        d_input.to_vec()
    }

    fn update_weights(&mut self, learning_rate: f64) {
        self.update_weights(learning_rate);
    }

    fn backward_batch(&mut self, grad_output: &[f64]) -> Vec<f64> {
        unimplemented!()
    }

    fn resize(&mut self, _input_size: usize, _output_size: usize) {
        unimplemented!()
    }

    fn assign_weights(&mut self, _other: &dyn TrainableLayer) {
        unimplemented!()
    }
    
    fn adjust_adam(
        &mut self,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        time_step: usize,
        moment1: &mut Matrix<f64>,
        moment2: &mut Matrix<f64>,
        bias_moment1: &mut Vec<f64>,
        bias_moment2: &mut Vec<f64>,
    ) {
        // Update kernel weights using Adam optimizer
        for row in 0..self.kernel.rows() {
            for col in 0..self.kernel.cols() {
                let grad = self.kernel.get_unchecked(row, col).grad;

                // Update moment estimates
                let m = beta1 * moment1.get_unchecked(row, col) + (1.0 - beta1) * grad;
                let v = beta2 * moment2.get_unchecked(row, col) + (1.0 - beta2) * grad * grad;

                // Bias correction
                let m_hat = m / (1.0 - beta1.powi(time_step as i32));
                let v_hat = v / (1.0 - beta2.powi(time_step as i32));

                // Update kernel value
                let new_value = self.kernel.get_unchecked(row, col).value - learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                self.kernel.get_mut_unchecked(row, col).value = new_value;

                // Update moments
                moment1.get_mut_unchecked(row, col).value = m;
                moment2.get_mut_unchecked(row, col).value = v;
            }
        }

        // Update biases using Adam optimizer
        for i in 0..self.biases.len() {
            let grad = self.biases[i]; // Assuming biases store gradients directly

            // Update moment estimates
            let m = beta1 * bias_moment1[i] + (1.0 - beta1) * grad;
            let v = beta2 * bias_moment2[i] + (1.0 - beta2) * grad * grad;

            // Bias correction
            let m_hat = m / (1.0 - beta1.powi(time_step as i32));
            let v_hat = v / (1.0 - beta2.powi(time_step as i32));

            // Update bias value
            self.biases[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);

            // Update moments
            bias_moment1[i] = m;
            bias_moment2[i] = v;
        }
    }
}
