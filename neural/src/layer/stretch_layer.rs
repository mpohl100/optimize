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

        let dense_layer_dims = DenseLayerDims::new(input_size, output_size);
        let dimensions = dense_layer_dims.get_dimensions();

        for (i, (dense_input_size, dense_output_size)) in dimensions.iter().enumerate() {
            let sub_directory = format!("dense_layer_{i}");
            let dense_layer_path = model_directory.expand(&sub_directory);
            let dense_layer = DenseLayer::new(
                *dense_input_size,
                *dense_output_size,
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
        let mut current_input_dim_start = 0;
        for (i, dense_layer) in self.dense_layers.iter_mut().enumerate() {
            let start = current_input_dim_start;
            let end = start + dense_layer.input_size();
            let input_slice = &input[start..end];
            let dense_output = dense_layer.forward(input_slice, utils.clone());
            for (j, &value) in dense_output.iter().enumerate() {
                let output_index = i * dense_layer.output_size() + j;
                if output_index < self.output_size {
                    output[output_index] = value;
                }
            }
            current_input_dim_start = end;
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
        // Initialize all values to zero
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                weights.set_mut_unchecked(i, j, NumberEntry::from(0.0));
            }
        }
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
        // Initialize all values to zero
        for i in 0..self.output_size {
            biases.set_mut_unchecked(i, 0, NumberEntry::from(0.0));
        }
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

        let dense_layer_dims = DenseLayerDims::new(input_size, output_size);
        let dimensions = dense_layer_dims.get_dimensions();

        for (i, (dense_input_size, dense_output_size)) in dimensions.iter().enumerate() {
            let sub_directory = format!("dense_layer_{i}");
            let dense_layer_path = model_directory.expand(&sub_directory);

            let dense_layer = TrainableDenseLayer::new(
                *dense_input_size,
                *dense_output_size,
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
        let mut current_input_dim_start = 0;
        for (i, dense_layer) in self.trainable_dense_layers.iter_mut().enumerate() {
            let start = current_input_dim_start;
            let end = start + dense_layer.input_size();
            let input_slice = &input[start..end];
            let dense_output = dense_layer.forward(input_slice, utils.clone());
            for (j, &value) in dense_output.iter().enumerate() {
                let output_index = i * dense_layer.output_size() + j;
                if output_index < self.output_size {
                    output[output_index] = value;
                } else {
                    panic!(
                        "Output index {} is out of bounds for output size {}",
                        output_index, self.output_size
                    );
                }
            }
            current_input_dim_start = end;
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
        // Initialize all values to zero
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                weights.set_mut_unchecked(i, j, WeightEntry::from(0.0));
            }
        }
        for (i, dense_layer) in self.trainable_dense_layers.iter().enumerate() {
            let dense_weights = dense_layer.get_weights();
            let top_left_row = i * dense_layer.output_size();
            let top_left_col = i * dense_layer.input_size();
            weights.set_submatrix(top_left_row, top_left_col, &dense_weights);
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
        // Initialize all values to zero
        for i in 0..self.output_size {
            biases.set_mut_unchecked(i, 0, BiasEntry::from(0.0));
        }
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
        let mut current_output_dim_start = 0;
        let mut current_input_dim_start = 0;
        for dense_layer in &mut self.trainable_dense_layers {
            // Slice d_out based on output dimensions
            let start = current_output_dim_start;
            let end = start + dense_layer.output_size();
            let d_vec_slice = &d_out[start..end];
            let dense_output = dense_layer.backward(d_vec_slice, utils.clone());
            // Accumulate gradients based on input dimensions;
            let input_start = current_input_dim_start;
            for (j, &value) in dense_output.iter().enumerate() {
                let ret_index = input_start + j;
                if ret_index < self.input_size {
                    ret[ret_index] += value;
                } else {
                    panic!(
                        "Ret index {} is out of bounds for input size {}",
                        ret_index, self.input_size
                    );
                }
            }
            current_output_dim_start = end;
            current_input_dim_start += dense_layer.input_size();
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

struct DenseLayerDims {
    dense_layer_dimensions: Vec<(usize, usize)>,
}

impl DenseLayerDims {
    pub fn new(
        input_size: usize,
        output_size: usize,
    ) -> Self {
        let (bigger_size, smaller_size) = if input_size >= output_size {
            (input_size, output_size)
        } else {
            (output_size, input_size)
        };
        let bigger_size_f64: f64 = NumCast::from(bigger_size).unwrap_or(0.0);
        let smaller_size_f64: f64 = NumCast::from(smaller_size).unwrap_or(0.0);
        let ratio = NumCast::from((bigger_size_f64 / smaller_size_f64).ceil()).unwrap_or(0);
        let num = NumCast::from(smaller_size).unwrap_or(0);
        let (dense_input_size, dense_output_size) =
            if input_size >= output_size { (ratio, 1) } else { (1, ratio) };
        let (dense_input_size_smaller, dense_output_size_smaller) =
            if dense_input_size >= dense_output_size { (ratio - 1, 1) } else { (1, ratio - 1) };

        let mut dense_layer_dimensions = Vec::new();
        let num_ratio = num - bigger_size % num;
        let num_ratio_minus_one = num - num_ratio;

        for _ in 0..num_ratio {
            dense_layer_dimensions.push((dense_input_size, dense_output_size));
        }

        for _ in 0..num_ratio_minus_one {
            dense_layer_dimensions.push((dense_input_size_smaller, dense_output_size_smaller));
        }

        Self { dense_layer_dimensions }
    }

    pub fn get_dimensions(&self) -> &Vec<(usize, usize)> {
        &self.dense_layer_dimensions
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
        // Use 10x10 to have a proper multi-layer structure
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

        // Get weights and biases from the first layer
        let weights1 = layer1.get_weights();
        let biases1 = layer1.get_biases();

        // Create a new trainable stretch layer with the same dimensions
        let mut layer2 = TrainableStretchLayer::new(
            10,
            10,
            Directory::Internal("test_transfer_stretch_trainable_2".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Assign the weights and biases from layer1 to layer2
        layer2.assign_trainable_layer(weights1.clone(), biases1.clone());

        // Get weights and biases from the second layer
        let weights2 = layer2.get_weights();
        let biases2 = layer2.get_biases();

        // Verify that all weights are correctly transferred
        assert_eq!(weights1.rows(), weights2.rows());
        assert_eq!(weights1.cols(), weights2.cols());
        println!("Checking weights: rows={}, cols={}", weights1.rows(), weights1.cols());
        let mut weight_mismatches = Vec::new();
        for i in 0..weights1.rows() {
            for j in 0..weights1.cols() {
                let val1 = weights1.get_unchecked(i, j);
                let val2 = weights2.get_unchecked(i, j);
                let diff = (val1.0.value - val2.0.value).abs();
                if diff > 1e-10 {
                    weight_mismatches.push((i, j, val1.0.value, val2.0.value, diff));
                    println!(
                        "Weight mismatch at [{}, {}]: val1={}, val2={}, diff={}",
                        i, j, val1.0.value, val2.0.value, diff
                    );
                }
            }
        }
        if !weight_mismatches.is_empty() {
            println!("Total weight mismatches: {}", weight_mismatches.len());
        }
        for i in 0..weights1.rows() {
            for j in 0..weights1.cols() {
                let val1 = weights1.get_unchecked(i, j);
                let val2 = weights2.get_unchecked(i, j);
                assert_abs_diff_eq!(val1.0.value, val2.0.value, epsilon = 1e-10);
            }
        }

        // Verify that all biases are correctly transferred
        assert_eq!(biases1.rows(), biases2.rows());
        println!("Checking biases: rows={}", biases1.rows());
        let mut bias_mismatches = Vec::new();
        for i in 0..biases1.rows() {
            let val1 = biases1.get_unchecked(i, 0);
            let val2 = biases2.get_unchecked(i, 0);
            let diff = (val1.0.value - val2.0.value).abs();
            if diff > 1e-10 {
                bias_mismatches.push((i, val1.0.value, val2.0.value, diff));
                println!(
                    "Bias mismatch at [{}]: val1={}, val2={}, diff={}",
                    i, val1.0.value, val2.0.value, diff
                );
            }
        }
        if !bias_mismatches.is_empty() {
            println!("Total bias mismatches: {}", bias_mismatches.len());
        }
        for i in 0..biases1.rows() {
            let val1 = biases1.get_unchecked(i, 0);
            let val2 = biases2.get_unchecked(i, 0);
            assert_abs_diff_eq!(val1.0.value, val2.0.value, epsilon = 1e-10);
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
        // Use 10x20 to have a proper multi-layer structure
        let mut trainable_layer = TrainableStretchLayer::new(
            10,
            20,
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
        let mut target = vec![0.0; 20];
        for i in 0..20 {
            target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        train_stretch_layer(&mut trainable_layer, &input, &target, &utils);

        // Get weights and biases from the trainable stretch layer
        let trainable_weights = trainable_layer.get_weights();
        let trainable_biases = trainable_layer.get_biases();

        // Create a StretchLayer (non-trainable)
        let mut stretch_layer = StretchLayer::new(
            10,
            20,
            Directory::Internal("test_conversion_stretch_dense".to_string()),
            0,
            MatrixParams { slice_rows: 10, slice_cols: 10 },
            &utils,
        );

        // Transfer weights from trainable to stretch layer
        stretch_layer.assign_trainable_layer(trainable_weights.clone(), trainable_biases.clone());

        // Get weights and biases from the stretch layer
        let stretch_weights = stretch_layer.get_weights();
        let stretch_biases = stretch_layer.get_biases();

        // Verify that all weights and biases are correctly transferred
        assert_eq!(trainable_weights.rows(), stretch_weights.rows());
        assert_eq!(trainable_weights.cols(), stretch_weights.cols());
        println!(
            "Checking weights: rows={}, cols={}",
            trainable_weights.rows(),
            trainable_weights.cols()
        );
        let mut weight_mismatches = Vec::new();
        for i in 0..trainable_weights.rows() {
            for j in 0..trainable_weights.cols() {
                let trainable_val = trainable_weights.get_unchecked(i, j);
                let stretch_val = stretch_weights.get_unchecked(i, j);
                let tv = trainable_val.0.value;
                let dv = stretch_val.0;
                let diff = (tv - dv).abs();
                if diff > 1e-10 {
                    weight_mismatches.push((i, j, tv, dv, diff));
                    println!(
                        "Weight mismatch at [{}, {}]: trainable={}, stretch={}, diff={}",
                        i, j, tv, dv, diff
                    );
                }
            }
        }
        if !weight_mismatches.is_empty() {
            println!("Total weight mismatches: {}", weight_mismatches.len());
        }
        for i in 0..trainable_weights.rows() {
            for j in 0..trainable_weights.cols() {
                let trainable_val = trainable_weights.get_unchecked(i, j);
                let stretch_val = stretch_weights.get_unchecked(i, j);
                let tv = trainable_val.0.value;
                let dv = stretch_val.0;
                assert_abs_diff_eq!(tv, dv, epsilon = 1e-10);
            }
        }

        // Verify that all biases are correctly transferred
        assert_eq!(trainable_biases.rows(), stretch_biases.rows());
        println!("Checking biases: rows={}", trainable_biases.rows());
        let mut bias_mismatches = Vec::new();
        for i in 0..trainable_biases.rows() {
            let trainable_val = trainable_biases.get_unchecked(i, 0);
            let stretch_val = stretch_biases.get_unchecked(i, 0);
            let tv = trainable_val.0.value;
            let dv = stretch_val.0;
            let diff = (tv - dv).abs();
            if diff > 1e-10 {
                bias_mismatches.push((i, tv, dv, diff));
                println!(
                    "Bias mismatch at [{}]: trainable={}, stretch={}, diff={}",
                    i, tv, dv, diff
                );
            }
        }
        if !bias_mismatches.is_empty() {
            println!("Total bias mismatches: {}", bias_mismatches.len());
        }
        for i in 0..trainable_biases.rows() {
            let trainable_val = trainable_biases.get_unchecked(i, 0);
            let stretch_val = stretch_biases.get_unchecked(i, 0);
            let tv = trainable_val.0.value;
            let dv = stretch_val.0;
            assert_abs_diff_eq!(tv, dv, epsilon = 1e-10);
        }

        // Cleanup
        trainable_layer.cleanup();
        stretch_layer.cleanup();
        let _ = std::fs::remove_dir_all("test_conversion_stretch_trainable");
        let _ = std::fs::remove_dir_all("test_conversion_stretch_dense");
    }
}
