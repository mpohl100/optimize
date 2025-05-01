use super::layer_trait::Layer;
use super::layer_trait::TrainableLayer;
use super::layer_trait::WrappedTrainableLayer;
use super::AllocatableLayer;
use super::TrainableAllocatableLayer;
use crate::alloc::allocatable::Allocatable;
pub use crate::neural::mat::matrix::Matrix;
use crate::neural::nn::directory::Directory;

use fs2::FileExt;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DenseLayer {
    rows: usize,
    cols: usize,
    weights: Option<Matrix<f64>>,
    biases: Option<Vec<f64>>,
    in_use: bool,
    layer_path: Directory,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
    ) -> Self {
        // create a Directory type which has the path model_directory/layers/layer_{position_in_nn}.txt
        let layer_path = match model_directory {
            Directory::User(path) => {
                Directory::User(format!("{}/layers/layer_{}.txt", path, position_in_nn))
            }
            Directory::Internal(path) => {
                Directory::Internal(format!("{}/layers/layer_{}.txt", path, position_in_nn))
            }
        };
        DenseLayer {
            rows: output_size,
            cols: input_size,
            weights: None,
            biases: None,
            in_use: false,
            layer_path,
        }
    }
}

impl Drop for DenseLayer {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        if let Directory::User(dir) = &self.layer_path {
            if std::fs::metadata(dir).is_ok() {
                // Save the model to disk
                self.deallocate();
            }
        }
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.layer_path {
            // check that dir is a file
            let path = Path::new(dir);
            // delete the file
            if path.is_file() {
                std::fs::remove_file(dir).expect("Failed to remove file");
            }
        }
    }
}

impl Allocatable for DenseLayer {
    fn allocate(&mut self) {
        if self.is_allocated() {
            return;
        }
        // if the layer_path does not exist, create a new matrix and store it
        if !self.layer_path.exists() {
            self.weights = Some(Matrix::new(self.rows, self.cols));
            self.biases = Some(vec![0.0; self.rows]);
            save(
                self.layer_path.path(),
                self.weights.as_ref().unwrap(),
                self.biases.as_ref().unwrap(),
            )
            .expect("Failed to save layer weights and biases");
        } else {
            // if the layer_path exists, read the matrix and store it
            let (weights, biases) =
                read(self.layer_path.path()).expect("Failed to read layer weights and biases");
            if self.rows == weights.rows() && self.cols == weights.cols() {
                self.rows = weights.rows();
                self.cols = weights.cols();
                self.weights = Some(weights);
                self.biases = Some(biases);
            } else {
                self.weights = Some(Matrix::new(self.rows, self.cols));
                self.biases = Some(vec![0.0; self.rows]);
                save(
                    self.layer_path.path(),
                    self.weights.as_ref().unwrap(),
                    self.biases.as_ref().unwrap(),
                )
                .expect("Failed to save layer weights and biases");
            }
        }
    }

    fn deallocate(&mut self) {
        if self.is_allocated() {
            save(
                self.layer_path.path(),
                self.weights.as_ref().unwrap(),
                self.biases.as_ref().unwrap(),
            )
            .expect("Failed to save layer weights and biases");
        }
        self.weights = None;
        self.biases = None;
    }

    fn is_allocated(&self) -> bool {
        self.weights.is_some() && self.biases.is_some()
    }

    fn get_size(&self) -> usize {
        (self.rows * self.cols + self.rows) * std::mem::size_of::<Weight>()
    }

    fn mark_for_use(&mut self) {
        self.in_use = true;
    }

    fn free_from_use(&mut self) {
        self.in_use = false;
    }

    fn is_in_use(&self) -> bool {
        self.in_use
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        if !self.is_allocated() {
            panic!("Layer not allocated");
        }
        let weights = self.weights.as_ref().unwrap();
        let biases = self.biases.as_ref().unwrap();
        weights
            .iter()
            .enumerate() // Include the row index in the iteration
            .map(|(row_idx, weights_row)| {
                weights_row
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + biases[row_idx] // Use the bias corresponding to the row index
            })
            .collect()
    }

    fn forward_batch(&mut self, _input: &[f64]) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.cols
    }

    fn output_size(&self) -> usize {
        self.rows
    }

    fn save(&self, path: String) -> Result<(), Box<dyn Error>> {
        save(
            path,
            self.weights.as_ref().unwrap(),
            self.biases.as_ref().unwrap(),
        )
    }

    fn read(&mut self, path: String) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read(path)?;
        self.rows = weights.rows();
        self.cols = weights.cols();
        self.weights = Some(weights);
        self.biases = Some(biases);
        Ok(())
    }

    fn get_weights(&self) -> Matrix<f64> {
        self.weights.as_ref().unwrap().clone()
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.as_ref().unwrap().clone()
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
}

impl AllocatableLayer for DenseLayer {
    fn duplicate(
        &mut self,
        model_directory: String,
        position_in_nn: usize,
    ) -> Box<dyn AllocatableLayer + Send> {
        self.deallocate();
        let new_layer = Box::new(DenseLayer::new(
            self.input_size(),
            self.output_size(),
            Directory::Internal(model_directory),
            position_in_nn,
        )) as Box<dyn AllocatableLayer + Send>;
        new_layer.copy_on_filesystem(self.layer_path.path().clone());
        new_layer
    }

    fn copy_on_filesystem(&self, layer_path: String) {
        // Copy the layer to the new directory
        let new_layer_path = self.layer_path.clone();
        let original_path = layer_path.clone();
        // Create the new directory if it doesn't exist
        let new_layer_path_string = new_layer_path.path();
        if !new_layer_path.exists() {
            // create the parent directory if it does not exist
            let p = Path::new(&new_layer_path_string);
            let parent_dir = p.parent().unwrap();
            std::fs::create_dir_all(parent_dir).expect("Failed to create directory");
        }
        let p_orig = Path::new(&original_path);
        if p_orig.is_file() {
            // copy the file
            std::fs::copy(original_path, new_layer_path.path()).expect("Failed to copy layer file");
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct Weight {
    value: f64,
    grad: f64,
    m: f64,
    v: f64,
}

#[derive(Default, Debug, Clone, Copy)]
struct Bias {
    value: f64,
    grad: f64,
    m: f64,
    v: f64,
}

/// A fully connected neural network layer (Dense layer).
#[derive(Debug, Clone)]
pub struct TrainableDenseLayer {
    rows: usize,
    cols: usize,
    weights: Option<Matrix<Weight>>, // Weight matrix (output_size x input_size)
    biases: Option<Vec<Bias>>,       // Bias vector (output_size)
    input_cache: Option<Vec<f64>>,   // Cache input for use in backward pass
    input_batch_cache: Option<Vec<Vec<f64>>>, // Cache batch input for use in backward pass
    in_use: bool,
    layer_path: Directory,
}

impl TrainableDenseLayer {
    /// Creates a new TrainableDenseLayer with given input and output sizes.
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
    ) -> Self {
        // create a Directory type which has the path model_directory/layers/layer_{position_in_nn}.txt
        let layer_path = match model_directory {
            Directory::User(path) => {
                Directory::User(format!("{}/layers/layer_{}.txt", path, position_in_nn))
            }
            Directory::Internal(path) => {
                Directory::Internal(format!("{}/layers/layer_{}.txt", path, position_in_nn))
            }
        };
        TrainableDenseLayer {
            rows: output_size,
            cols: input_size,
            weights: None,
            biases: None,
            input_cache: None,
            input_batch_cache: None,
            in_use: false,
            layer_path,
        }
    }

    /// Initialize the weights with random values in the range [-0.5, 0.5]
    fn initialize_weights(&mut self) {
        let mut rng = rand::thread_rng();
        // initialize weights from -0.5 to 0.5
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                self.weights.as_mut().unwrap().get_mut_unchecked(i, j).value =
                    rng.gen_range(-0.5..0.5);
            }
        }
    }
}

impl Drop for TrainableDenseLayer {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        if let Directory::User(dir) = &self.layer_path {
            if std::fs::metadata(dir).is_ok() {
                // Save the model to disk
                self.deallocate();
            }
        }
    }
}

impl Allocatable for TrainableDenseLayer {
    fn allocate(&mut self) {
        if self.is_allocated() {
            return;
        }
        // if the layer_path does not exist, create a new matrix and store it
        if !self.layer_path.exists() {
            self.weights = Some(Matrix::new(self.rows, self.cols));
            self.biases = Some(vec![Bias::default(); self.rows]);
            self.initialize_weights();
            save_weight(
                self.layer_path.path(),
                self.weights.as_ref().unwrap(),
                self.biases.as_ref().unwrap(),
            )
            .expect("Failed to save layer weights and biases");
        } else {
            // if the layer_path exists, read the matrix and store it
            let (weights, biases) = read_weight(self.layer_path.path())
                .expect("Failed to read layer weights and biases");
            if self.rows == weights.rows() && self.cols == weights.cols() {
                self.rows = weights.rows();
                self.cols = weights.cols();
                self.weights = Some(weights);
                self.biases = Some(biases);
            } else {
                self.weights = Some(Matrix::new(self.rows, self.cols));
                self.biases = Some(vec![Bias::default(); self.rows]);
                self.initialize_weights();
                save_weight(
                    self.layer_path.path(),
                    self.weights.as_ref().unwrap(),
                    self.biases.as_ref().unwrap(),
                )
                .expect("Failed to save layer weights and biases");
            }
        }
        self.input_cache = Some(Vec::new());
        self.input_batch_cache = Some(Vec::new());
    }

    fn deallocate(&mut self) {
        if self.is_allocated() {
            save_weight(
                self.layer_path.path(),
                self.weights.as_ref().unwrap(),
                self.biases.as_ref().unwrap(),
            )
            .expect("Failed to save layer weights and biases");
        }
        self.weights = None;
        self.biases = None;
        self.input_cache = None;
        self.input_batch_cache = None;
    }

    fn is_allocated(&self) -> bool {
        self.weights.is_some() && self.biases.is_some()
    }

    fn get_size(&self) -> usize {
        (self.rows * self.cols + self.rows) * std::mem::size_of::<Weight>()
    }

    fn mark_for_use(&mut self) {
        self.in_use = true;
    }

    fn free_from_use(&mut self) {
        self.in_use = false;
    }

    fn is_in_use(&self) -> bool {
        self.in_use
    }
}

impl Layer for TrainableDenseLayer {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        if !self.is_allocated() {
            panic!("Layer not allocated");
        }
        self.input_cache = Some(input.to_vec()); // Cache the input for backpropagation
        let weights = self.weights.as_ref().unwrap();
        let biases = self.biases.as_ref().unwrap();
        weights
            .iter()
            .enumerate() // Include the row index in the iteration
            .map(|(row_idx, weights_row)| {
                weights_row
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w.value * x)
                    .sum::<f64>()
                    + biases[row_idx].value // Use the bias corresponding to the row index
            })
            .collect()
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_batch(&mut self, _input: &[f64]) -> Vec<f64> {
        unimplemented!()
    }

    fn input_size(&self) -> usize {
        self.cols
    }

    fn output_size(&self) -> usize {
        self.rows
    }

    fn save(&self, path: String) -> Result<(), Box<dyn Error>> {
        if !self.is_allocated() {
            // just copy the files
            let original_path = self.layer_path.path();
            // if the original path does not exist early return
            if std::fs::metadata(original_path.clone()).is_err() {
                return Ok(());
            }
            let file_path = Path::new(&path);
            if file_path.is_file() && original_path != path {
                // copy the file
                std::fs::copy(original_path, path).expect("Failed to copy file in save layer");
            }
            return Ok(());
        }
        // assign weights and biases to a matrix and vector
        let mut weights = Matrix::new(
            self.weights.as_ref().unwrap().rows(),
            self.weights.as_ref().unwrap().cols(),
        );
        let mut biases = vec![0.0; self.biases.as_ref().unwrap().len()];
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                *weights.get_mut_unchecked(i, j) =
                    self.weights.as_ref().unwrap().get_unchecked(i, j).value;
            }
        }
        for (i, bias) in self.biases.as_ref().unwrap().iter().enumerate() {
            biases[i] = bias.value;
        }
        save(path, &weights, &biases)
    }

    fn read(&mut self, path: String) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read(path)?;
        self.rows = weights.rows();
        self.cols = weights.cols();
        self.allocate();
        // assign all weights and biases
        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    self.weights.as_mut().unwrap().get_mut_unchecked(i, j).value =
                        *weights.get_unchecked(i, j);
                }
            }
            if i < biases.len() {
                self.biases.as_mut().unwrap()[i].value = biases[i];
            }
        }
        Ok(())
    }

    fn get_weights(&self) -> Matrix<f64> {
        let mut weights = Matrix::new(
            self.weights.as_ref().unwrap().rows(),
            self.weights.as_ref().unwrap().cols(),
        );
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                *weights.get_mut_unchecked(i, j) =
                    self.weights.as_ref().unwrap().get_unchecked(i, j).value;
            }
        }
        weights
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases
            .as_ref()
            .unwrap()
            .iter()
            .map(|bias| bias.value)
            .collect()
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
}

impl TrainableLayer for TrainableDenseLayer {
    /// Backward pass for the dense layer
    ///
    /// - `d_out`: Gradient of the loss with respect to the output of this layer
    /// - Returns: Gradient of the loss with respect to the input of this layer
    fn backward(&mut self, d_out: &[f64]) -> Vec<f64> {
        // Calculate weight gradients
        for (i, row_grad) in self.weights.as_mut().unwrap().iter_mut().enumerate() {
            for (j, grad) in row_grad.iter_mut().enumerate() {
                grad.grad = d_out[i] * self.input_cache.as_ref().unwrap()[j];
            }
        }

        // Calculate input gradients
        let mut d_input = vec![0.0; self.input_cache.as_ref().unwrap().len()];
        for (i, weights_row) in self.weights.as_ref().unwrap().iter().enumerate() {
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
        if !self.is_allocated() {
            panic!("Layer not allocated");
        }
        // Update weight
        for weights_row in self.weights.as_mut().unwrap().iter_mut() {
            for weight in weights_row.iter_mut() {
                weight.value -= learning_rate * weight.grad;
            }
        }

        // Update biases
        for bias in self.biases.as_mut().unwrap().iter_mut() {
            bias.value -= learning_rate * bias.grad;
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn backward_batch(&mut self, _grad_output: &[f64]) -> Vec<f64> {
        unimplemented!()
    }

    fn assign_weights(&mut self, other: WrappedTrainableLayer) {
        let weights = other.get_weights();

        let biases = other.get_biases();

        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                if i < weights.rows() && j < weights.cols() {
                    self.weights.as_mut().unwrap().get_mut_unchecked(i, j).value =
                        *weights.get_unchecked(i, j);
                }
            }
            if i < biases.len() {
                self.biases.as_mut().unwrap()[i].value = biases[i];
            }
        }
    }

    fn adjust_adam(&mut self, t: usize, learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) {
        // Update weights
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                let grad = self.weights.as_ref().unwrap().get_unchecked(i, j).grad;

                // Update first and second moments
                self.weights.as_mut().unwrap().get_mut_unchecked(i, j).m = beta1
                    * self.weights.as_ref().unwrap().get_unchecked(i, j).m
                    + (1.0 - beta1) * grad;
                self.weights.as_mut().unwrap().get_mut_unchecked(i, j).v = beta2
                    * self.weights.as_ref().unwrap().get_unchecked(i, j).v
                    + (1.0 - beta2) * grad.powi(2);

                // Bias correction
                let m_hat = self.weights.as_ref().unwrap().get_unchecked(i, j).m
                    / (1.0 - beta1.powi(t as i32));
                let v_hat = self.weights.as_ref().unwrap().get_unchecked(i, j).v
                    / (1.0 - beta2.powi(t as i32));

                // Adjusted learning rate
                let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);

                // Update weights
                self.weights.as_mut().unwrap().get_mut_unchecked(i, j).value -=
                    adjusted_learning_rate * m_hat;
            }
        }

        // Update biases
        for i in 0..self.biases.as_ref().unwrap().len() {
            let grad = self.biases.as_ref().unwrap()[i].grad;

            // Update first and second moments
            self.biases.as_mut().unwrap()[i].m =
                beta1 * self.biases.as_ref().unwrap()[i].m + (1.0 - beta1) * grad;
            self.biases.as_mut().unwrap()[i].v =
                beta2 * self.biases.as_ref().unwrap()[i].v + (1.0 - beta2) * grad.powi(2);

            // Bias correction
            let m_hat = self.biases.as_ref().unwrap()[i].m / (1.0 - beta1.powi(t as i32));
            let v_hat = self.biases.as_ref().unwrap()[i].v / (1.0 - beta2.powi(t as i32));

            // Adjusted learning rate
            let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);

            // Update biases
            self.biases.as_mut().unwrap()[i].value -= adjusted_learning_rate * m_hat;
        }
    }

    fn save_weight(&self, path: String) -> Result<(), Box<dyn Error>> {
        if !self.is_allocated() {
            // just copy the files
            let original_path = self.layer_path.path();
            // if the original path does not exist early return
            if std::fs::metadata(original_path.clone()).is_err() {
                return Ok(());
            }
            let file_path = Path::new(&path);
            if file_path.is_file() && original_path != path {
                std::fs::copy(original_path, path)
                    .expect("Failed to copy file in save layer weight");
            }
            return Ok(());
        }
        // assign weights and biases to a matrix and vector
        let mut weights = Matrix::new(
            self.weights.as_ref().unwrap().rows(),
            self.weights.as_ref().unwrap().cols(),
        );
        let mut biases = vec![Bias::default(); self.biases.as_ref().unwrap().len()];
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                *weights.get_mut_unchecked(i, j) =
                    *self.weights.as_ref().unwrap().get_unchecked(i, j);
            }
        }
        for (i, bias) in self.biases.as_ref().unwrap().iter().enumerate() {
            biases[i] = *bias;
        }
        save_weight(path, &weights, &biases)
    }

    fn read_weight(&mut self, path: String) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read_weight(path)?;
        self.rows = weights.rows();
        self.cols = weights.cols();
        self.allocate();
        // assign all weights and biases
        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    *self.weights.as_mut().unwrap().get_mut_unchecked(i, j) =
                        *weights.get_unchecked(i, j);
                }
            }
            if i < biases.len() {
                self.biases.as_mut().unwrap()[i] = biases[i];
            }
        }
        Ok(())
    }
}

impl TrainableAllocatableLayer for TrainableDenseLayer {
    fn duplicate(
        &mut self,
        model_directory: String,
        position_in_nn: usize,
    ) -> Box<dyn TrainableAllocatableLayer + Send> {
        self.deallocate();
        let new_layer = Box::new(TrainableDenseLayer::new(
            self.input_size(),
            self.output_size(),
            Directory::Internal(model_directory),
            position_in_nn,
        )) as Box<dyn TrainableAllocatableLayer + Send>;
        new_layer.copy_on_filesystem(self.layer_path.path().clone());
        new_layer
    }

    fn copy_on_filesystem(&self, layer_path: String) {
        // Copy the layer to the new directory
        let new_layer_path = self.layer_path.clone();
        let original_path = layer_path.clone();
        // Create the new directory if it doesn't exist
        let new_layer_path_string = new_layer_path.path();
        if !new_layer_path.exists() {
            // create the parent directory if it does not exist
            let p = Path::new(&new_layer_path_string);
            let parent_dir = p.parent().unwrap();
            std::fs::create_dir_all(parent_dir).expect("Failed to create directory");
        }
        // Copy the file of the original path to the new path on the filesystem
        let p_orig = Path::new(&original_path);
        if p_orig.is_file() {
            // copy the file
            std::fs::copy(original_path, new_layer_path.path()).expect("Failed to copy layer file");
        }
    }
}

fn save(path: String, weights: &Matrix<f64>, biases: &[f64]) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    let p = Path::new(&path);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).expect("Failed to create directory");
    }
    // create a lock file which acts as a lock
    let lock_file_path = format!("{}.lock", path);
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;
    // Save weights and biases to a file at the specified path
    let mut file = File::create(path)?;
    writeln!(file, "{} {}", weights.rows(), weights.cols())?;
    for i in 0..weights.rows() {
        for j in 0..weights.cols() {
            write!(file, "{};", weights.get_unchecked(i, j))?;
        }
        writeln!(file)?;
    }
    for bias in biases.iter() {
        write!(file, "{} ", bias)?;
    }
    writeln!(file)?;
    Ok(())
}

fn save_weight(
    path: String,
    weights: &Matrix<Weight>,
    biases: &[Bias],
) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    let p = Path::new(&path);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).expect("Failed to create directory");
    }

    // create a lock file which acts as a lock
    let lock_file_path = format!("{}.lock", path);
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    // Save weights and biases to a file at the specified path
    let mut file = File::create(path)?;
    writeln!(file, "{} {}", weights.rows(), weights.cols())?;
    for i in 0..weights.rows() {
        for j in 0..weights.cols() {
            let weight = weights.get_unchecked(i, j);
            write!(
                file,
                "{} {} {} {};",
                weight.value, weight.grad, weight.m, weight.v
            )?;
        }
        writeln!(file)?;
    }
    for bias in biases.iter() {
        write!(file, "{} {} {} {};", bias.value, bias.grad, bias.m, bias.v)?;
    }
    writeln!(file)?;
    Ok(())
}

fn read(path: String) -> Result<(Matrix<f64>, Vec<f64>), Box<dyn Error>> {
    // create a lock file which acts as a lock
    let lock_file_path = format!("{}.lock", path);
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

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
                let parts = line.split(";").collect::<Vec<_>>();
                // parts len must be euqal to cols
                if parts.len() - 1 != cols {
                    return Err(format!(
                        "Invalid weight format cause of cols: expected {}, found {}",
                        cols,
                        parts.len() - 1
                    )
                    .into());
                }
                for j in 0..cols {
                    let part = parts.get(j);
                    if let Some(p) = part {
                        let figures = p.split_whitespace().collect::<Vec<_>>();
                        if figures.len() == 1 || figures.len() == 4 {
                            *weights.get_mut_unchecked(i, j) = figures[0].parse::<f64>()?;
                        } else {
                            return Err("Invalid weight format".into());
                        };
                    }
                }
            }
        }
    }
    if let Some(Ok(line)) = lines.next() {
        let parts = line.split(";").collect::<Vec<_>>();
        biases = vec![0.0; weights.rows()];
        // parts len must be equal to rows
        if parts.len() - 1 != weights.rows() {
            return Err(format!(
                "Invalid bias format amount of values: expected {}, found {}",
                weights.rows(),
                parts.len() - 1
            )
            .into());
        }
        for (i, bias) in biases.iter_mut().enumerate() {
            let part = parts.get(i);
            if let Some(p) = part {
                let figures = p.split_whitespace().collect::<Vec<_>>();
                if figures.len() == 1 || figures.len() == 4 {
                    *bias = figures[0].parse::<f64>()?;
                } else {
                    return Err("Invalid bias format".into());
                }
            }
        }
    }
    Ok((weights, biases))
}

fn read_weight(path: String) -> Result<(Matrix<Weight>, Vec<Bias>), Box<dyn Error>> {
    // create a lock file which acts as a lock
    let lock_file_path = format!("{}.lock", path);
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut weights = Matrix::new(1, 1);
    let mut biases = vec![Bias::default(); 1];
    if let Some(Ok(line)) = lines.next() {
        let mut dim_parts = line.split_whitespace();
        let rows = dim_parts.next().unwrap().parse::<usize>()?;
        let cols = dim_parts.next().unwrap().parse::<usize>()?;
        weights = Matrix::new(rows, cols);
        for i in 0..rows {
            if let Some(Ok(line)) = lines.next() {
                let parts = line.split(";").collect::<Vec<_>>();
                // parts len must be euqal to cols
                if parts.len() - 1 != cols {
                    return Err(format!(
                        "Invalid weight format cause of cols: expected {}, found {}",
                        cols,
                        parts.len() - 1
                    )
                    .into());
                }
                for j in 0..cols {
                    let part = parts.get(j);
                    if let Some(p) = part {
                        let figures = p.split_whitespace().collect::<Vec<_>>();
                        if figures.len() == 4 {
                            *weights.get_mut_unchecked(i, j) = Weight {
                                value: figures[0].parse::<f64>()?,
                                grad: figures[1].parse::<f64>()?,
                                m: figures[2].parse::<f64>()?,
                                v: figures[3].parse::<f64>()?,
                            }
                        } else if figures.len() == 1 {
                            *weights.get_mut_unchecked(i, j) = Weight {
                                value: figures[0].parse::<f64>()?,
                                grad: 0.0,
                                m: 0.0,
                                v: 0.0,
                            };
                        } else {
                            return Err("Invalid weight format".into());
                        };
                    }
                }
            }
        }
    }
    if let Some(Ok(line)) = lines.next() {
        let parts = line.split(";").collect::<Vec<_>>();
        biases = vec![Bias::default(); weights.rows()];
        // parts len must be equal to rows
        if parts.len() - 1 != weights.rows() {
            return Err(format!(
                "Invalid bias format amount of values: expected {}, found {}",
                weights.rows(),
                parts.len() - 1
            )
            .into());
        }
        for (i, bias) in biases.iter_mut().enumerate() {
            let part = parts.get(i);
            if let Some(p) = part {
                let figures = p.split_whitespace().collect::<Vec<_>>();
                if figures.len() == 4 {
                    bias.value = figures[0].parse::<f64>()?;
                    bias.grad = figures[1].parse::<f64>()?;
                    bias.m = figures[2].parse::<f64>()?;
                    bias.v = figures[3].parse::<f64>()?;
                } else if figures.len() == 1 {
                    bias.value = figures[0].parse::<f64>()?;
                    bias.grad = 0.0;
                    bias.m = 0.0;
                    bias.v = 0.0;
                } else {
                    return Err("Invalid bias format".into());
                }
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
        let mut layer =
            TrainableDenseLayer::new(3, 2, Directory::Internal("test_model_unit".to_string()), 0);

        let input = vec![1.0, 2.0, 3.0];
        layer.allocate();
        layer.mark_for_use();
        let output = layer.forward(&input);
        layer.free_from_use();

        assert_eq!(output.len(), 2);

        let grad_output = vec![0.1, 0.2];
        let grad_input = layer.backward(&grad_output);

        assert_eq!(grad_input.len(), 3);

        layer.update_weights(0.01);

        std::fs::remove_dir_all("test_model_unit").unwrap();
    }
}
