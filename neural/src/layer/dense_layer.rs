use super::layer_trait::Layer;
use super::layer_trait::TrainableLayer;
use crate::layer::matrix_extensions::MatrixExtensionsWrappedComposite;
use crate::layer::matrix_extensions::TrainableMatrixExtensionsWrappedComposite;
use crate::utilities::util::WrappedUtils;
use matrix::ai_types::Bias;
use matrix::ai_types::BiasEntry;
use matrix::ai_types::Weight;
use matrix::ai_types::WeightEntry;
use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::directory::Directory;

use matrix::ai_types::NumberEntry;
pub use matrix::mat::Matrix;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

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
    weights: WrappedCompositeMatrix<NumberEntry>,
    biases: WrappedCompositeMatrix<NumberEntry>,
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
        utils: &WrappedUtils,
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
            utils.get_matrix_alloc_manager(),
        ));
        let biases = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            matrix_params.slice_cols,
            output_size,
            1,
            &layer_path.expand("biases"),
            utils.get_matrix_alloc_manager(),
        ));
        Self { rows: output_size, cols: input_size, weights, biases, layer_path }
    }
}

impl Layer<NumberEntry, NumberEntry> for DenseLayer {
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        let weights = self.weights.clone();
        let biases = self.biases.clone();
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
        self.weights.save()?;
        self.biases.save()?;
        Ok(())
    }

    fn read(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn get_weights(&self) -> WrappedCompositeMatrix<NumberEntry> {
        self.weights.clone()
    }

    fn get_biases(&self) -> WrappedCompositeMatrix<NumberEntry> {
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

    fn assign_layer(
        &mut self,
        weights: WrappedCompositeMatrix<NumberEntry>,
        biases: WrappedCompositeMatrix<NumberEntry>,
    ) {
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    self.weights.set_mut_unchecked(i, j, NumberEntry(v.0));
                }
            }
            if i < biases.rows() {
                let b = biases.get_unchecked(i, 0);
                self.biases.set_mut_unchecked(i, 0, NumberEntry(b.0));
            }
        }
    }

    fn assign_trainable_layer(
        &mut self,
        weights: WrappedCompositeMatrix<WeightEntry>,
        biases: WrappedCompositeMatrix<BiasEntry>,
    ) {
        for i in 0..self.weights.rows() {
            for j in 0..self.weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    self.weights.set_mut_unchecked(i, j, NumberEntry(v.0.value));
                }
            }
            if i < biases.rows() {
                let b = biases.get_unchecked(i, 0);
                self.biases.set_mut_unchecked(i, 0, NumberEntry(b.0.value));
            }
        }
    }
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
        utils: &WrappedUtils,
    ) -> Self {
        // create a Directory type which has the path model_directory/layers/layer_{position_in_nn}.txt
        let layer_path = model_directory.expand(&format!("layer_{position_in_nn}"));
        let weights = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            matrix_params.slice_cols,
            output_size,
            input_size,
            &layer_path.expand("weights"),
            utils.get_trainable_weight_matrix_alloc_manager(),
        ));
        let biases = WrappedCompositeMatrix::new(CompositeMatrix::new(
            matrix_params.slice_rows,
            matrix_params.slice_cols,
            output_size,
            1,
            &layer_path.expand("biases"),
            utils.get_trainable_bias_matrix_alloc_manager(),
        ));
        Self {
            rows: output_size,
            cols: input_size,
            weights,
            biases,
            input_cache: None,
            _input_batch_cache: None,
            layer_path,
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

    fn assign_layer(
        &mut self,
        weights: WrappedCompositeMatrix<NumberEntry>,
        biases: WrappedCompositeMatrix<NumberEntry>,
    ) {
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

    fn assign_trainable_layer(
        &mut self,
        weights: WrappedCompositeMatrix<WeightEntry>,
        biases: WrappedCompositeMatrix<BiasEntry>,
    ) {
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
            &utils,
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

    /// Helper function to compare two weight matrices
    fn weight_matrices_equal(
        a: &WrappedCompositeMatrix<WeightEntry>,
        b: &WrappedCompositeMatrix<WeightEntry>,
    ) -> bool {
        if a.rows() != b.rows() || a.cols() != b.cols() {
            return false;
        }
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                let val_a = a.get_unchecked(i, j);
                let val_b = b.get_unchecked(i, j);
                if val_a.0.value != val_b.0.value {
                    println!(
                        "Weight mismatch at ({}, {}): {} != {}",
                        i, j, val_a.0.value, val_b.0.value
                    );
                    return false;
                }
            }
        }
        true
    }

    /// Helper function to compare two bias matrices
    fn bias_matrices_equal(
        a: &WrappedCompositeMatrix<BiasEntry>,
        b: &WrappedCompositeMatrix<BiasEntry>,
    ) -> bool {
        if a.rows() != b.rows() || a.cols() != b.cols() {
            return false;
        }
        for i in 0..a.rows() {
            for j in 0..a.cols() {
                let val_a = a.get_unchecked(i, j);
                let val_b = b.get_unchecked(i, j);
                if val_a.0.value != val_b.0.value {
                    println!(
                        "Bias mismatch at ({}, {}): {} != {}",
                        i, j, val_a.0.value, val_b.0.value
                    );
                    return false;
                }
            }
        }
        true
    }

    /// Helper function to train a simple trainable dense layer
    fn train_simple_layer(
        layer: &mut TrainableDenseLayer,
        utils: WrappedUtils,
        epochs: usize,
    ) {
        // Simple training loop
        let input = vec![0.5; 10];
        let target = vec![0.8; 10];

        for _ in 0..epochs {
            let output = layer.forward(&input, utils.clone());
            let mut grad = vec![0.0; 10];
            for i in 0..10 {
                grad[i] = output[i] - target[i];
            }
            layer.backward(&grad, utils.clone());
            layer.update_weights(0.01, utils.clone());
        }
    }

    #[test]
    fn test_trainable_layer_weight_transfer_to_new_trainable_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));

        // Create and train first trainable layer
        let mut layer1 = TrainableDenseLayer::new(
            10,
            10,
            &Directory::Internal("test_transfer_trainable_1".to_string()),
            0,
            MatrixParams { slice_rows: 3, slice_cols: 3 },
            &utils,
        );

        // Train the layer
        train_simple_layer(&mut layer1, utils.clone(), 10);

        // Get weights and biases from the first layer
        let weights1 = layer1.get_weights();
        let biases1 = layer1.get_biases();

        // Create a new trainable layer with the same dimensions
        let mut layer2 = TrainableDenseLayer::new(
            10,
            10,
            &Directory::Internal("test_transfer_trainable_2".to_string()),
            0,
            MatrixParams { slice_rows: 3, slice_cols: 3 },
            &utils,
        );

        // Assign the weights and biases from layer1 to layer2
        layer2.assign_trainable_layer(weights1.clone(), biases1.clone());

        // Get weights and biases from the second layer
        let weights2 = layer2.get_weights();
        let biases2 = layer2.get_biases();

        // Verify that all values are correctly transferred
        assert!(
            weight_matrices_equal(&weights1, &weights2),
            "Weights were not correctly transferred"
        );
        assert!(bias_matrices_equal(&biases1, &biases2), "Biases were not correctly transferred");

        // Cleanup
        layer1.cleanup();
        layer2.cleanup();
        let _ = std::fs::remove_dir_all("test_transfer_trainable_1");
        let _ = std::fs::remove_dir_all("test_transfer_trainable_2");
    }

    #[test]
    fn test_trainable_layer_conversion_to_dense_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));

        // Create and train a trainable layer
        let mut trainable_layer = TrainableDenseLayer::new(
            10,
            10,
            &Directory::Internal("test_conversion_trainable".to_string()),
            0,
            MatrixParams { slice_rows: 3, slice_cols: 3 },
            &utils,
        );

        // Train the layer
        train_simple_layer(&mut trainable_layer, utils.clone(), 10);

        // Get weights and biases from the trainable layer
        let trainable_weights = trainable_layer.get_weights();
        let trainable_biases = trainable_layer.get_biases();

        // Create a DenseLayer (non-trainable)
        let mut dense_layer = DenseLayer::new(
            10,
            10,
            Directory::Internal("test_conversion_dense".to_string()),
            0,
            MatrixParams { slice_rows: 3, slice_cols: 3 },
            &utils,
        );

        // Transfer weights from trainable to dense layer
        dense_layer.assign_trainable_layer(trainable_weights.clone(), trainable_biases.clone());

        // Get weights and biases from the dense layer
        let dense_weights = dense_layer.get_weights();
        let dense_biases = dense_layer.get_biases();

        // Verify that all weights and biases are correctly transferred
        // Compare the values (trainable layer stores Weight/Bias structs, dense layer stores f64 values)
        assert_eq!(trainable_weights.rows(), dense_weights.rows());
        assert_eq!(trainable_weights.cols(), dense_weights.cols());
        assert_eq!(trainable_biases.rows(), dense_biases.rows());

        for i in 0..trainable_weights.rows() {
            for j in 0..trainable_weights.cols() {
                let trainable_val = trainable_weights.get_unchecked(i, j);
                let dense_val = dense_weights.get_unchecked(i, j);
                assert_eq!(
                    trainable_val.0.value, dense_val.0,
                    "Weight mismatch at ({}, {}): {} != {}",
                    i, j, trainable_val.0.value, dense_val.0
                );
            }
        }

        for i in 0..trainable_biases.rows() {
            let trainable_val = trainable_biases.get_unchecked(i, 0);
            let dense_val = dense_biases.get_unchecked(i, 0);
            assert_eq!(
                trainable_val.0.value, dense_val.0,
                "Bias mismatch at {}: {} != {}",
                i, trainable_val.0.value, dense_val.0
            );
        }

        // Cleanup
        trainable_layer.cleanup();
        dense_layer.cleanup();
        let _ = std::fs::remove_dir_all("test_conversion_trainable");
        let _ = std::fs::remove_dir_all("test_conversion_dense");
    }

    #[test]
    fn test_trainable_layer_save_and_load_as_dense_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));

        // Create a trainable layer
        let trainable_base_dir = Directory::Internal("test_save_load_final".to_string());

        let mut trainable_layer = TrainableDenseLayer::new(
            10,
            10,
            &trainable_base_dir,
            0,
            MatrixParams { slice_rows: 3, slice_cols: 3 },
            &utils,
        );

        // Train the layer
        train_simple_layer(&mut trainable_layer, utils.clone(), 10);

        // Save the trainable layer
        trainable_layer.save(String::new()).expect("Failed to save trainable layer");

        // Get the weights from the trainable layer before it's dropped
        let trainable_weights = trainable_layer.get_weights();
        let trainable_biases = trainable_layer.get_biases();

        // Create a DenseLayer in a different directory
        let dense_base_dir = Directory::Internal("test_save_load_dense_final".to_string());
        let mut dense_layer = DenseLayer::new(
            10,
            10,
            dense_base_dir,
            0,
            MatrixParams { slice_rows: 3, slice_cols: 3 },
            &utils,
        );

        // Transfer the weights from the saved trainable layer to the dense layer
        // This simulates reading a saved trainable layer and converting it to a dense layer
        dense_layer.assign_trainable_layer(trainable_weights.clone(), trainable_biases.clone());

        // Get the weights from the dense layer
        let dense_weights = dense_layer.get_weights();
        let dense_biases = dense_layer.get_biases();

        // Verify that all weights and biases are correctly transferred
        assert_eq!(trainable_weights.rows(), dense_weights.rows());
        assert_eq!(trainable_weights.cols(), dense_weights.cols());
        assert_eq!(trainable_biases.rows(), dense_biases.rows());

        for i in 0..trainable_weights.rows() {
            for j in 0..trainable_weights.cols() {
                let trainable_val = trainable_weights.get_unchecked(i, j);
                let dense_val = dense_weights.get_unchecked(i, j);
                assert_eq!(
                    trainable_val.0.value, dense_val.0,
                    "Weight mismatch at ({}, {}): {} != {}",
                    i, j, trainable_val.0.value, dense_val.0
                );
            }
        }

        for i in 0..trainable_biases.rows() {
            let trainable_val = trainable_biases.get_unchecked(i, 0);
            let dense_val = dense_biases.get_unchecked(i, 0);
            assert_eq!(
                trainable_val.0.value, dense_val.0,
                "Bias mismatch at {}: {} != {}",
                i, trainable_val.0.value, dense_val.0
            );
        }

        // Save the dense layer to verify the save API works
        dense_layer.save(String::new()).expect("Failed to save dense layer");

        // Cleanup
        trainable_layer.cleanup();
        dense_layer.cleanup();
        let _ = std::fs::remove_dir_all("test_save_load_final");
        let _ = std::fs::remove_dir_all("test_save_load_dense_final");
    }
}
