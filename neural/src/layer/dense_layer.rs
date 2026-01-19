use super::layer_trait::Layer;
use super::layer_trait::TrainableLayer;
use super::layer_trait::WrappedTrainableLayer;
use crate::layer::layer_trait::WrappedLayer;
use crate::layer::matrix_extensions::MatrixExtensionsWrappedComposite;
use crate::layer::matrix_extensions::TrainableMatrixExtensionsWrappedComposite;
use crate::utilities::util::WrappedUtils;
use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::directory::Directory;

pub use matrix::mat::Matrix;

use matrix::persistable_matrix::PersistableValue;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use rand::Rng;
use serde::Deserialize;
use serde::Serialize;
use std::error::Error;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Copy)]
pub struct MatrixParams {
    pub slice_rows: usize,
    pub slice_cols: usize,
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    rows: usize,
    cols: usize,
    weights_new: WrappedCompositeMatrix<NumberEntry>,
    biases_new: WrappedCompositeMatrix<NumberEntry>,
    layer_path: Directory,
}

impl DenseLayer {
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
        matrix_params: MatrixParams,
    ) -> Self {
        // create a Directory type which has the path model_directory/layers/layer_{position_in_nn}.txt
        let layer_path = match model_directory {
            Directory::User(path) => {
                Directory::User(format!("{path}/layers/layer_{position_in_nn}.txt"))
            },
            Directory::Internal(path) => {
                Directory::Internal(format!("{path}/layers/layer_{position_in_nn}.txt"))
            },
        };
        let weights = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            matrix_params.slice_cols,
            output_size,
            input_size,
            &layer_path.expand("weights"),
        ));
        let biases = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            1,
            output_size,
            1,
            &layer_path.expand("biases"),
        ));
        Self {
            rows: output_size,
            cols: input_size,
            weights_new: weights,
            biases_new: biases,
            layer_path,
        }
    }
}

impl Layer<NumberEntry, NumberEntry> for DenseLayer {
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        let weights = self.weights_new.clone();
        let biases = self.biases_new.clone();
        let inputs = input.to_vec();
        utils.execute(move || weights.forward(&inputs, &biases))
    }

    fn forward_batch(
        &mut self,
        _input: &[f64],
    ) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.cols
    }

    fn output_size(&self) -> usize {
        self.rows
    }

    fn save(
        &self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        self.weights_new.save()?;
        self.biases_new.save()?;
        Ok(())
    }

    fn read(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn get_weights(&self) -> WrappedCompositeMatrix<NumberEntry> {
        self.weights_new.clone()
    }

    fn get_biases(&self) -> WrappedCompositeMatrix<NumberEntry> {
        self.biases_new.clone()
    }

    fn cleanup(&self) {
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.layer_path {
            // delete file
            if std::fs::metadata(dir).is_ok() {
                // check that dir is a file
                let path = Path::new(dir);
                // delete the file
                if path.is_file() {
                    std::fs::remove_file(dir).expect("Failed to remove file");
                }
            }
        }
    }

    fn assign_weights(
        &mut self,
        other: WrappedLayer<NumberEntry, NumberEntry>,
    ) {
        let weights = other.get_weights();
        let biases = other.get_biases();
        for i in 0..self.weights_new.rows() {
            for j in 0..self.weights_new.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    self.weights_new.set_mut_unchecked(i, j, NumberEntry(v.0));
                }
            }
            if i < biases.rows() {
                let b = biases.get_unchecked(i, 0);
                self.biases_new.set_mut_unchecked(i, 0, NumberEntry(b.0));
            }
        }
    }

    fn assign_trainable_weights(
        &mut self,
        other: WrappedTrainableLayer<WeightEntry, BiasEntry>,
    ) {
        let weights = other.get_weights();
        let biases = other.get_biases();
        for i in 0..self.weights_new.rows() {
            for j in 0..self.weights_new.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    self.weights_new.set_mut_unchecked(i, j, NumberEntry(v.0.value));
                }
            }
            if i < biases.rows() {
                let b = biases.get_unchecked(i, 0);
                self.biases_new.set_mut_unchecked(i, 0, NumberEntry(b.0.value));
            }
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Weight {
    pub value: f64,
    pub grad: f64,
    pub m: f64,
    pub v: f64,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Bias {
    pub value: f64,
    pub grad: f64,
    pub m: f64,
    pub v: f64,
}

/// A fully connected neural network layer (Dense layer).
#[derive(Debug, Clone)]
pub struct TrainableDenseLayer {
    rows: usize,
    cols: usize,
    weights: WrappedCompositeMatrix<WeightEntry>, // Weight matrix (output_size x input_size)
    biases: WrappedCompositeMatrix<BiasEntry>,    // Bias vector (output_size)
    input_cache: Option<Vec<f64>>,                // Cache input for use in backward pass
    _input_batch_cache: Option<Vec<Vec<f64>>>,    // Cache batch input for use in backward pass
    layer_path: Directory,
}

impl TrainableDenseLayer {
    /// Creates a new `TrainableDenseLayer` with given input and output sizes.
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: &Directory,
        position_in_nn: usize,
        matrix_params: MatrixParams,
    ) -> Self {
        // create a Directory type which has the path model_directory/layers/layer_{position_in_nn}.txt
        let layer_path = model_directory.expand(&format!("layer_{position_in_nn}"));
        let weights = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            matrix_params.slice_cols,
            output_size,
            input_size,
            &layer_path.expand("weights"),
        ));
        let biases = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            1,
            output_size,
            1,
            &layer_path.expand("biases"),
        ));
        let layer = Self {
            rows: output_size,
            cols: input_size,
            weights,
            biases,
            input_cache: None,
            _input_batch_cache: None,
            layer_path,
        };
        layer.initialize_weights();
        layer
    }

    /// Initialize the weights with random values in the range [-0.5, 0.5]
    fn initialize_weights(&self) {
        let mut rng = rand::thread_rng();
        // initialize weights from -0.5 to 0.5
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                let value = rng.gen_range(-0.5..0.5);
                let w = Weight { value, grad: 0.0, m: 0.0, v: 0.0 };
                self.weights.set_mut_unchecked(i, j, WeightEntry(w));
            }
        }
    }
}

impl Layer<WeightEntry, BiasEntry> for TrainableDenseLayer {
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        self.input_cache = Some(input.to_vec()); // Cache the input for backpropagation
        let weights = self.weights.clone();
        let biases = self.biases.clone();
        let inputs = input.to_vec();
        utils.execute(move || weights.forward(&inputs, &biases))
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_batch(
        &mut self,
        _input: &[f64],
    ) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.cols
    }

    fn output_size(&self) -> usize {
        self.rows
    }

    fn save(
        &self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        // assign weights and biases to a matrix and vector
        self.weights.save()?;
        self.biases.save()?;
        Ok(())
    }

    fn read(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        Ok(())
    }

    fn get_weights(&self) -> WrappedCompositeMatrix<WeightEntry> {
        self.weights.clone()
    }

    fn get_biases(&self) -> WrappedCompositeMatrix<BiasEntry> {
        self.biases.clone()
    }

    fn cleanup(&self) {
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.layer_path {
            // delete file
            if std::fs::metadata(dir).is_ok() {
                // check that dir is a file
                let path = Path::new(dir);
                // delete the file
                if path.is_file() {
                    std::fs::remove_file(dir).expect("Failed to remove file");
                }
            }
        }
    }

    fn assign_weights(
        &mut self,
        other: WrappedLayer<NumberEntry, NumberEntry>,
    ) {
        let weights = other.get_weights();
        let biases = other.get_biases();
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    let w = Weight { value: v.0, grad: 0.0, m: 0.0, v: 0.0 };
                    self.weights.set_mut_unchecked(i, j, WeightEntry(w));
                }
            }
            if i < biases.rows() {
                let b = biases.get_unchecked(i, 0);
                let bias = Bias { value: b.0, grad: 0.0, m: 0.0, v: 0.0 };
                self.biases.set_mut_unchecked(i, 0, BiasEntry(bias));
            }
        }
    }

    fn assign_trainable_weights(
        &mut self,
        other: WrappedTrainableLayer<WeightEntry, BiasEntry>,
    ) {
        let weights = other.get_weights();
        let biases = other.get_biases();
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    self.weights.set_mut_unchecked(i, j, WeightEntry(v.0));
                }
            }
            if i < biases.rows() {
                let b = biases.get_unchecked(i, 0);
                self.biases.set_mut_unchecked(i, 0, BiasEntry(b.0));
            }
        }
    }
}

impl TrainableLayer<WeightEntry, BiasEntry> for TrainableDenseLayer {
    /// Backward pass for the dense layer
    ///
    /// - `d_out`: Gradient of the loss with respect to the output of this layer
    /// - Returns: Gradient of the loss with respect to the input of this layer
    fn backward(
        &mut self,
        d_out: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        let weights = self.weights.clone();
        let input_cache = self.input_cache.as_ref().unwrap().clone();
        let d_out_vec = d_out.to_vec();
        // Calculate weight gradients
        let _ = utils.execute(move || {
            weights.backward_calculate_gradients(&d_out_vec, &input_cache);
            0
        });

        let weights_sec = self.weights.clone();
        let input_cache_sec = self.input_cache.as_ref().unwrap().clone();
        let d_out_vec_sec = d_out.to_vec();
        // Calculate input gradients
        utils.execute(move || {
            (0..input_cache_sec.len())
                .into_par_iter()
                .map(|j| weights_sec.backward_calculate_weights_sec(j, &d_out_vec_sec))
                .collect::<Vec<f64>>()
        })
    }

    /// Update weights and biases using their respective gradients
    ///
    /// - `learning_rate`: The step size for gradient descent
    fn update_weights(
        &mut self,
        learning_rate: f64,
        utils: WrappedUtils,
    ) {
        let weights = self.weights.clone();
        // Update weight
        let _ = utils.execute(move || {
            weights.update_weights(learning_rate);
            0
        });

        // Update biases
        let biases = self.biases.clone();
        let _ = utils.execute(move || {
            biases.update_weights(learning_rate);
            0
        });
    }

    #[allow(clippy::needless_range_loop)]
    fn backward_batch(
        &mut self,
        _grad_output: &[f64],
    ) -> Vec<f64> {
        unimplemented!()
    }

    fn adjust_adam(
        &mut self,
        t: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        utils: WrappedUtils,
    ) {
        let weights = self.weights.clone();
        // Update weights
        let _ = utils.execute(move || {
            weights.adjust_adam(beta1, beta2, epsilon, t, learning_rate);
            0
        });

        let biases = self.biases.clone();
        // Update biases
        let _ = utils.execute(move || {
            biases.adjust_adam(beta1, beta2, epsilon, t, learning_rate);
            0
        });
    }

    fn save_weight(
        &self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        self.weights.save()?;
        self.biases.save()?;
        Ok(())
    }

    fn read_weight(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct NumberEntry(pub f64);

impl PersistableValue for NumberEntry {
    fn to_string_for_matrix(&self) -> String {
        format!("{}", self.0)
    }

    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let value = s.parse::<f64>()?;
        Ok(Self(value))
    }
}

#[derive(Debug, Default, Clone)]
pub struct WeightEntry(pub Weight);

impl PersistableValue for WeightEntry {
    fn to_string_for_matrix(&self) -> String {
        format!("{} {} {} {}", self.0.value, self.0.grad, self.0.m, self.0.v)
    }

    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 4 {
            return Err("Invalid weight entry format".into());
        }
        let value = parts[0].parse::<f64>()?;
        let grad = parts[1].parse::<f64>()?;
        let m = parts[2].parse::<f64>()?;
        let v = parts[3].parse::<f64>()?;
        Ok(Self(Weight { value, grad, m, v }))
    }
}

#[derive(Debug, Default, Clone)]
pub struct BiasEntry(pub Bias);

impl PersistableValue for BiasEntry {
    fn to_string_for_matrix(&self) -> String {
        format!("{} {} {} {}", self.0.value, self.0.grad, self.0.m, self.0.v)
    }

    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 4 {
            return Err("Invalid bias entry format".into());
        }
        let value = parts[0].parse::<f64>()?;
        let grad = parts[1].parse::<f64>()?;
        let m = parts[2].parse::<f64>()?;
        let v = parts[3].parse::<f64>()?;
        Ok(Self(Bias { value, grad, m, v }))
    }
}

#[cfg(test)]
mod tests {
    use crate::utilities::util::Utils;

    use super::*;

    #[test]
    fn test_dense_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer = TrainableDenseLayer::new(
            3,
            2,
            &Directory::Internal("test_model_unit".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
        );

        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input, utils.clone());

        assert_eq!(output.len(), 2);

        let grad_output = vec![0.1, 0.2];
        let grad_input = layer.backward(&grad_output, utils.clone());

        assert_eq!(grad_input.len(), 3);

        layer.update_weights(0.01, utils);

        std::fs::remove_dir_all("test_model_unit").unwrap();
    }
}
