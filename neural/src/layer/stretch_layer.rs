use crate::layer::dense_layer::DenseLayer;
use crate::layer::dense_layer::MatrixParams;
use crate::layer::dense_layer::TrainableDenseLayer;
use crate::layer::layer_trait::Layer;
use crate::layer::layer_trait::TrainableLayer;
use crate::utilities::util::WrappedUtils;

use crate::layer::matrix_extensions::BiasEntry;
use crate::layer::matrix_extensions::WeightEntry;
use alloc::alloc_manager::WrappedAllocManager;
use matrix::ai_types::NumberEntry;
use matrix::composite_matrix::CompositeMatrix;
use matrix::composite_matrix::WrappedCompositeMatrix;
use matrix::directory::Directory;
use matrix::persistable_matrix::WrappedPersistableMatrix;
use num_traits::NumCast;

use std::error::Error;

#[derive(Debug, Clone)]
pub struct StretchLayer {
    input_size: usize,
    output_size: usize,
    dense_layers: Vec<DenseLayer>,
    layer_path: Directory,
    matrix_params: MatrixParams,
    alloc_manager: WrappedAllocManager<WrappedPersistableMatrix<NumberEntry>>,
}

impl StretchLayer {
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
        matrix_params: MatrixParams,
        utils: &WrappedUtils,
    ) -> Self {
        let mut dense_layers = Vec::new();

        let input_size_f64: f64 = NumCast::from(input_size).unwrap_or(0.0);
        let output_size_f64: f64 = NumCast::from(output_size).unwrap_or(0.0);
        let num;
        let (dense_input_size, dense_output_size) = if input_size >= output_size {
            num = NumCast::from((input_size_f64 / output_size_f64).ceil()).unwrap_or(0);
            (num, 1)
        } else {
            num = NumCast::from((output_size_f64 / input_size_f64).ceil()).unwrap_or(0);
            (1, num)
        };

        for i in 0..num {
            let sub_directory = format!("dense_layer_{i}");
            let dense_layer_path = model_directory.expand(&sub_directory);
            let dense_layer = DenseLayer::new(
                dense_input_size,
                dense_output_size,
                &dense_layer_path,
                position_in_nn,
                matrix_params,
                utils,
            );
            dense_layers.push(dense_layer);
        }
        Self {
            input_size,
            output_size,
            dense_layers,
            layer_path: model_directory,
            matrix_params,
            alloc_manager: utils.get_matrix_alloc_manager(),
        }
    }
}

impl Layer<NumberEntry, NumberEntry> for StretchLayer {
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for (i, dense_layer) in self.dense_layers.iter_mut().enumerate() {
            let start = i * dense_layer.input_size();
            let end = start + dense_layer.input_size();
            let input_slice = &input[start..end];
            let dense_output = dense_layer.forward(input_slice, utils.clone());
            for (j, &value) in dense_output.iter().enumerate() {
                let output_index = i * dense_layer.output_size() + j;
                if output_index < self.output_size {
                    output[output_index] = value;
                }
            }
        }
        output
    }

    fn forward_batch(
        &mut self,
        _input: &[f64],
    ) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn save(
        &self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        for (i, dense_layer) in self.dense_layers.iter().enumerate() {
            dense_layer.save(format!("dense_layer_{i}"))?;
        }
        Ok(())
    }

    fn read(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        for (i, dense_layer) in self.dense_layers.iter_mut().enumerate() {
            dense_layer.read(format!("dense_layer_{i}"))?;
        }
        Ok(())
    }

    fn get_weights(&self) -> WrappedCompositeMatrix<NumberEntry> {
        let internal_mat_directory = self.layer_path.expand("temp_weights").to_internal();
        let weights = WrappedCompositeMatrix::new(CompositeMatrix::new(
            self.matrix_params.slice_rows,
            self.matrix_params.slice_cols,
            self.output_size,
            self.input_size,
            &internal_mat_directory,
            self.alloc_manager.clone(),
        ));
        for (i, dense_layer) in self.dense_layers.iter().enumerate() {
            let dense_weights = dense_layer.get_weights();
            weights.set_submatrix(
                i * dense_layer.output_size(),
                i * dense_layer.input_size(),
                &dense_weights,
            );
        }

        weights
    }

    fn get_biases(&self) -> WrappedCompositeMatrix<NumberEntry> {
        let internal_mat_directory = self.layer_path.expand("temp_biases").to_internal();
        let biases = WrappedCompositeMatrix::new(CompositeMatrix::new(
            self.matrix_params.slice_rows,
            self.matrix_params.slice_cols,
            self.output_size,
            1,
            &internal_mat_directory,
            self.alloc_manager.clone(),
        ));
        for (i, dense_layer) in self.dense_layers.iter().enumerate() {
            let dense_biases = dense_layer.get_biases();
            biases.set_submatrix(i * dense_layer.output_size(), 0, &dense_biases);
        }

        biases
    }

    fn cleanup(&self) {
        // Remove the internal model directory from disk
        for dense_layer in &self.dense_layers {
            dense_layer.cleanup();
        }
    }

    fn assign_layer(
        &mut self,
        weights: WrappedCompositeMatrix<NumberEntry>,
        biases: WrappedCompositeMatrix<NumberEntry>,
    ) {
        for (i, dense_layer) in self.dense_layers.iter_mut().enumerate() {
            let dense_weights = weights.get_submatrix(
                i * dense_layer.output_size(),
                i * dense_layer.input_size(),
                dense_layer.output_size(),
                dense_layer.input_size(),
            );
            let dense_biases = biases.get_submatrix(
                i * dense_layer.output_size(),
                0,
                dense_layer.output_size(),
                1,
            );
            dense_layer.assign_layer(dense_weights, dense_biases);
        }
    }

    fn assign_trainable_layer(
        &mut self,
        weights: WrappedCompositeMatrix<WeightEntry>,
        biases: WrappedCompositeMatrix<BiasEntry>,
    ) {
        for (i, dense_layer) in self.dense_layers.iter_mut().enumerate() {
            let dense_trainable_weights = weights.get_submatrix(
                i * dense_layer.output_size(),
                i * dense_layer.input_size(),
                dense_layer.output_size(),
                dense_layer.input_size(),
            );
            let dense_trainable_biases = biases.get_submatrix(
                i * dense_layer.output_size(),
                0,
                dense_layer.output_size(),
                1,
            );
            dense_layer.assign_trainable_layer(dense_trainable_weights, dense_trainable_biases);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainableStretchLayer {
    input_size: usize,
    output_size: usize,
    trainable_dense_layers: Vec<TrainableDenseLayer>,
    layer_path: Directory,
    matrix_params: MatrixParams,
    weight_alloc_manager: WrappedAllocManager<WrappedPersistableMatrix<WeightEntry>>,
    bias_alloc_manager: WrappedAllocManager<WrappedPersistableMatrix<BiasEntry>>,
}

impl TrainableStretchLayer {
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
        matrix_params: MatrixParams,
        utils: &WrappedUtils,
    ) -> Self {
        let mut dense_layers = Vec::new();

        let input_size_f64: f64 = NumCast::from(input_size).unwrap_or(0.0);
        let output_size_f64: f64 = NumCast::from(output_size).unwrap_or(0.0);
        let num;
        let (dense_input_size, dense_output_size) = if input_size >= output_size {
            num = NumCast::from((input_size_f64 / output_size_f64).ceil()).unwrap_or(0);
            (num, 1)
        } else {
            num = NumCast::from((output_size_f64 / input_size_f64).ceil()).unwrap_or(0);
            (1, num)
        };

        for i in 0..num {
            let sub_directory = format!("dense_layer_{i}");
            let dense_layer_path = model_directory.expand(&sub_directory);
            let dense_layer = TrainableDenseLayer::new(
                dense_input_size,
                dense_output_size,
                &dense_layer_path,
                position_in_nn,
                matrix_params,
                utils,
            );
            dense_layers.push(dense_layer);
        }
        Self {
            input_size,
            output_size,
            trainable_dense_layers: dense_layers,
            layer_path: model_directory,
            matrix_params,
            weight_alloc_manager: utils.get_trainable_weight_matrix_alloc_manager(),
            bias_alloc_manager: utils.get_trainable_bias_matrix_alloc_manager(),
        }
    }
}

impl Layer<WeightEntry, BiasEntry> for TrainableStretchLayer {
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for (i, dense_layer) in self.trainable_dense_layers.iter_mut().enumerate() {
            let start = i * dense_layer.input_size();
            let end = start + dense_layer.input_size();
            let input_slice = &input[start..end];
            let dense_output = dense_layer.forward(input_slice, utils.clone());
            for (j, &value) in dense_output.iter().enumerate() {
                let output_index = i * dense_layer.output_size() + j;
                if output_index < self.output_size {
                    output[output_index] = value;
                }
            }
        }
        output
    }

    fn forward_batch(
        &mut self,
        _input: &[f64],
    ) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn save(
        &self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        for (i, dense_layer) in self.trainable_dense_layers.iter().enumerate() {
            dense_layer.save(format!("dense_layer_{i}"))?;
        }
        Ok(())
    }

    fn read(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        for (i, dense_layer) in self.trainable_dense_layers.iter_mut().enumerate() {
            dense_layer.read(format!("dense_layer_{i}"))?;
        }
        Ok(())
    }

    fn get_weights(&self) -> WrappedCompositeMatrix<WeightEntry> {
        let internal_mat_directory = self.layer_path.expand("temp_weights").to_internal();
        let weights = WrappedCompositeMatrix::new(CompositeMatrix::new(
            self.matrix_params.slice_rows,
            self.matrix_params.slice_cols,
            self.output_size,
            self.input_size,
            &internal_mat_directory,
            self.weight_alloc_manager.clone(),
        ));
        for (i, dense_layer) in self.trainable_dense_layers.iter().enumerate() {
            let dense_weights = dense_layer.get_weights();
            weights.set_submatrix(
                i * dense_layer.output_size(),
                i * dense_layer.input_size(),
                &dense_weights,
            );
        }

        weights
    }

    fn get_biases(&self) -> WrappedCompositeMatrix<BiasEntry> {
        let internal_mat_directory = self.layer_path.expand("temp_biases").to_internal();
        let biases = WrappedCompositeMatrix::new(CompositeMatrix::new(
            self.matrix_params.slice_rows,
            self.matrix_params.slice_cols,
            self.output_size,
            1,
            &internal_mat_directory,
            self.bias_alloc_manager.clone(),
        ));
        for (i, dense_layer) in self.trainable_dense_layers.iter().enumerate() {
            let dense_biases = dense_layer.get_biases();
            biases.set_submatrix(i * dense_layer.output_size(), 0, &dense_biases);
        }

        biases
    }

    fn cleanup(&self) {
        // Remove the internal model directory from disk
        for dense_layer in &self.trainable_dense_layers {
            dense_layer.cleanup();
        }
    }

    fn assign_layer(
        &mut self,
        weights: WrappedCompositeMatrix<NumberEntry>,
        biases: WrappedCompositeMatrix<NumberEntry>,
    ) {
        for (i, dense_layer) in self.trainable_dense_layers.iter_mut().enumerate() {
            let dense_weights = weights.get_submatrix(
                i * dense_layer.output_size(),
                i * dense_layer.input_size(),
                dense_layer.output_size(),
                dense_layer.input_size(),
            );
            let dense_biases = biases.get_submatrix(
                i * dense_layer.output_size(),
                0,
                dense_layer.output_size(),
                1,
            );
            dense_layer.assign_layer(dense_weights, dense_biases);
        }
    }

    fn assign_trainable_layer(
        &mut self,
        weights: WrappedCompositeMatrix<WeightEntry>,
        biases: WrappedCompositeMatrix<BiasEntry>,
    ) {
        for (i, dense_layer) in self.trainable_dense_layers.iter_mut().enumerate() {
            let dense_trainable_weights = weights.get_submatrix(
                i * dense_layer.output_size(),
                i * dense_layer.input_size(),
                dense_layer.output_size(),
                dense_layer.input_size(),
            );
            let dense_trainable_biases = biases.get_submatrix(
                i * dense_layer.output_size(),
                0,
                dense_layer.output_size(),
                1,
            );
            dense_layer.assign_trainable_layer(dense_trainable_weights, dense_trainable_biases);
        }
    }
}

impl TrainableLayer<WeightEntry, BiasEntry> for TrainableStretchLayer {
    /// Backward pass for the dense layer
    ///
    /// - `d_out`: Gradient of the loss with respect to the output of this layer
    /// - Returns: Gradient of the loss with respect to the input of this layer
    fn backward(
        &mut self,
        d_out: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        /*
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
        */
        let mut ret = vec![0.0; self.input_size];
        for (i, dense_layer) in self.trainable_dense_layers.iter_mut().enumerate() {
            // Slice d_out based on output dimensions
            let start = i * dense_layer.output_size();
            let end = start + dense_layer.output_size();
            let d_vec_slice = &d_out[start..end];
            let dense_output = dense_layer.backward(d_vec_slice, utils.clone());
            // Accumulate gradients based on input dimensions
            for (j, &value) in dense_output.iter().enumerate() {
                let ret_index = i * dense_layer.input_size() + j;
                if ret_index < self.input_size {
                    ret[ret_index] += value;
                }
            }
        }
        ret
    }

    /// Update weights and biases using their respective gradients
    ///
    /// - `learning_rate`: The step size for gradient descent
    fn update_weights(
        &mut self,
        learning_rate: f64,
        utils: WrappedUtils,
    ) {
        for dense_layer in &mut self.trainable_dense_layers {
            dense_layer.update_weights(learning_rate, utils.clone());
        }
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
        for dense_layer in &mut self.trainable_dense_layers {
            dense_layer.adjust_adam(t, learning_rate, beta1, beta2, epsilon, utils.clone());
        }
    }

    fn save_weight(
        &self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        for dense_layer in &self.trainable_dense_layers {
            dense_layer.save_weight(format!("dense_layer_{}", dense_layer.output_size()))?;
        }
        Ok(())
    }

    fn read_weight(
        &mut self,
        _path: String,
    ) -> Result<(), Box<dyn Error>> {
        for dense_layer in &mut self.trainable_dense_layers {
            dense_layer.read_weight(format!("dense_layer_{}", dense_layer.output_size()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::util::Utils;
    use approx::assert_abs_diff_eq;

    /// Helper function to train a stretch layer with given input and target for 5 epochs
    fn train_stretch_layer(
        layer: &mut TrainableStretchLayer,
        input: &[f64],
        target: &[f64],
        utils: &WrappedUtils,
    ) {
        let epochs = 5;
        for _ in 0..epochs {
            let output = layer.forward(input, utils.clone());
            let mut grad = vec![0.0; output.len()];
            for i in 0..grad.len() {
                grad[i] = output[i] - target[i];
            }
            layer.backward(&grad, utils.clone());
            layer.update_weights(0.01, utils.clone());
        }
    }

    #[test]
    fn test_stretch_layer_50x1_to_10x1() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer = TrainableStretchLayer::new(
            50,
            10,
            Directory::Internal("test_stretch_50_10".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Create input with alternating 1s and 0s
        let mut input = vec![0.0; 50];
        for i in 0..50 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Target with alternating pattern
        let mut target = vec![0.0; 10];
        for i in 0..10 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        train_stretch_layer(&mut layer, &input, &target, &utils);

        // Verify the output dimensions are correct
        let output = layer.forward(&input, utils.clone());
        assert_eq!(output.len(), 10);

        layer.cleanup();
        let _ = std::fs::remove_dir_all("test_stretch_50_10");
    }

    #[test]
    fn test_stretch_layer_25x1_to_10x1() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer = TrainableStretchLayer::new(
            25,
            10,
            Directory::Internal("test_stretch_25_10".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Create input with alternating 1s and 0s
        let mut input = vec![0.0; 25];
        for i in 0..25 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Target with alternating pattern
        let mut target = vec![0.0; 10];
        for i in 0..10 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        train_stretch_layer(&mut layer, &input, &target, &utils);

        // Verify the output dimensions are correct
        let output = layer.forward(&input, utils.clone());
        assert_eq!(output.len(), 10);

        layer.cleanup();
        let _ = std::fs::remove_dir_all("test_stretch_25_10");
    }

    #[test]
    fn test_stretch_layer_10x1_to_10x1() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer = TrainableStretchLayer::new(
            10,
            10,
            Directory::Internal("test_stretch_10_10".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Create input with alternating 1s and 0s
        let mut input = vec![0.0; 10];
        for i in 0..10 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Target should be the same as input (no transformation needed)
        let mut target = vec![0.0; 10];
        for i in 0..10 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        train_stretch_layer(&mut layer, &input, &target, &utils);

        // Verify the output dimensions are correct
        let output = layer.forward(&input, utils.clone());
        assert_eq!(output.len(), 10);

        layer.cleanup();
        let _ = std::fs::remove_dir_all("test_stretch_10_10");
    }

    #[test]
    fn test_stretch_layer_10x1_to_25x1() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer = TrainableStretchLayer::new(
            10,
            25,
            Directory::Internal("test_stretch_10_25".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Create input with alternating 1s and 0s
        let mut input = vec![0.0; 10];
        for i in 0..10 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Target with alternating pattern
        let mut target = vec![0.0; 25];
        for i in 0..25 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        train_stretch_layer(&mut layer, &input, &target, &utils);

        // Verify the output dimensions are correct
        let output = layer.forward(&input, utils.clone());
        assert_eq!(output.len(), 25);

        layer.cleanup();
        let _ = std::fs::remove_dir_all("test_stretch_10_25");
    }

    #[test]
    fn test_stretch_layer_10x1_to_50x1() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer = TrainableStretchLayer::new(
            10,
            50,
            Directory::Internal("test_stretch_10_50".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Create input with alternating 1s and 0s
        let mut input = vec![0.0; 10];
        for i in 0..10 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Target with alternating pattern
        let mut target = vec![0.0; 50];
        for i in 0..50 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }

        train_stretch_layer(&mut layer, &input, &target, &utils);

        // Verify the output dimensions are correct
        let output = layer.forward(&input, utils.clone());
        assert_eq!(output.len(), 50);

        layer.cleanup();
        let _ = std::fs::remove_dir_all("test_stretch_10_50");
    }

    #[test]
    fn test_trainable_stretch_layer_weight_transfer_to_new_trainable_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));

        // Create and train first trainable stretch layer
        let mut layer1 = TrainableStretchLayer::new(
            10,
            10,
            Directory::Internal("test_transfer_stretch_trainable_1".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Train the layer
        let mut input = vec![0.0; 10];
        for i in 0..10 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        let mut target = vec![0.0; 10];
        for i in 0..10 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        train_stretch_layer(&mut layer1, &input, &target, &utils);

        // Get output from the trained layer
        let output1 = layer1.forward(&input, utils.clone());

        // Create a new trainable stretch layer with the same dimensions
        let mut layer2 = TrainableStretchLayer::new(
            10,
            10,
            Directory::Internal("test_transfer_stretch_trainable_2".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Transfer weights by copying each internal dense layer's weights
        for (i, (src_layer, dst_layer)) in layer1
            .trainable_dense_layers
            .iter()
            .zip(layer2.trainable_dense_layers.iter_mut())
            .enumerate()
        {
            let weights = src_layer.get_weights();
            let biases = src_layer.get_biases();
            dst_layer.assign_trainable_layer(weights, biases);
            println!("Transferred weights for dense layer {}", i);
        }

        // Get output from the second layer (should match layer1)
        let output2 = layer2.forward(&input, utils.clone());

        // Verify that outputs match (weights were successfully transferred)
        assert_eq!(output1.len(), output2.len());
        for i in 0..output1.len() {
            assert_abs_diff_eq!(output1[i], output2[i], epsilon = 1e-10);
        }

        // Cleanup
        layer1.cleanup();
        layer2.cleanup();
        let _ = std::fs::remove_dir_all("test_transfer_stretch_trainable_1");
        let _ = std::fs::remove_dir_all("test_transfer_stretch_trainable_2");
    }

    #[test]
    fn test_trainable_stretch_layer_conversion_to_stretch_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));

        // Create and train a trainable stretch layer
        let mut trainable_layer = TrainableStretchLayer::new(
            10,
            10,
            Directory::Internal("test_conversion_stretch_trainable".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Train the layer
        let mut input = vec![0.0; 10];
        for i in 0..10 {
            input[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        let mut target = vec![0.0; 10];
        for i in 0..10 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        train_stretch_layer(&mut trainable_layer, &input, &target, &utils);

        // Get output from the trained layer
        let output_trainable = trainable_layer.forward(&input, utils.clone());

        // Create a StretchLayer (non-trainable)
        let mut stretch_layer = StretchLayer::new(
            10,
            10,
            Directory::Internal("test_conversion_stretch_dense".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Transfer weights by copying each internal dense layer's weights
        for (i, (src_layer, dst_layer)) in trainable_layer
            .trainable_dense_layers
            .iter()
            .zip(stretch_layer.dense_layers.iter_mut())
            .enumerate()
        {
            let weights = src_layer.get_weights();
            let biases = src_layer.get_biases();
            dst_layer.assign_trainable_layer(weights, biases);
            println!("Transferred weights for dense layer {}", i);
        }

        // Get output from the stretch layer (should match trainable layer)
        let output_stretch = stretch_layer.forward(&input, utils.clone());

        // Verify that outputs match (weights were successfully transferred)
        assert_eq!(output_trainable.len(), output_stretch.len());
        for i in 0..output_trainable.len() {
            assert_abs_diff_eq!(output_trainable[i], output_stretch[i], epsilon = 1e-10);
        }

        // Cleanup
        trainable_layer.cleanup();
        stretch_layer.cleanup();
        let _ = std::fs::remove_dir_all("test_conversion_stretch_trainable");
        let _ = std::fs::remove_dir_all("test_conversion_stretch_dense");
    }
}
