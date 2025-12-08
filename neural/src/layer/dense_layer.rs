use super::layer_trait::Layer;
use super::layer_trait::TrainableLayer;
use super::layer_trait::WrappedTrainableLayer;
use super::matrix_extensions::MatrixExtensions;
use super::AllocatableLayer;
use super::TrainableAllocatableLayer;
use crate::layer::matrix_extensions::TrainableMatrixExtensions;
use crate::utilities::util::WrappedUtils;
use alloc::allocatable::Allocatable;
use matrix::directory::Directory;

pub use matrix::mat::Matrix;
use matrix::mat::WrappedMatrix;

use matrix::persistable_matrix::PersistableValue;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use num_traits::cast::NumCast;

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
    weights: Option<WrappedMatrix<f64>>,
    biases: Option<Vec<f64>>,
    in_use: bool,
    layer_path: Directory,
}

impl DenseLayer {
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
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
        Self {
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
        if self.layer_path.exists() {
            // if the layer_path exists, read the matrix and store it
            let (weights, biases) =
                read(self.layer_path.path()).expect("Failed to read layer weights and biases");
            if self.rows == weights.rows() && self.cols == weights.cols() {
                self.rows = weights.rows();
                self.cols = weights.cols();
                self.weights = Some(weights);
                self.biases = Some(biases);
            } else {
                self.weights = Some(WrappedMatrix::new(self.rows, self.cols));
                self.biases = Some(vec![0.0; self.rows]);
                save(
                    self.layer_path.path(),
                    self.weights.as_ref().unwrap(),
                    self.biases.as_ref().unwrap(),
                )
                .expect("Failed to save layer weights and biases");
            }
        } else {
            self.weights = Some(WrappedMatrix::new(self.rows, self.cols));
            self.biases = Some(vec![0.0; self.rows]);
            save(
                self.layer_path.path(),
                self.weights.as_ref().unwrap(),
                self.biases.as_ref().unwrap(),
            )
            .expect("Failed to save layer weights and biases");
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
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        assert!(self.is_allocated(), "Layer not allocated");
        let weights = self.weights.as_ref().unwrap().clone();
        let biases = self.biases.as_ref().unwrap().clone();
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
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        save(path, self.weights.as_ref().unwrap(), self.biases.as_ref().unwrap())
    }

    fn read(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read(path)?;
        self.rows = weights.rows();
        self.cols = weights.cols();
        self.weights = Some(weights);
        self.biases = Some(biases);
        Ok(())
    }

    fn get_weights(&self) -> WrappedMatrix<f64> {
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
        let new_layer = Box::new(Self::new(
            self.input_size(),
            self.output_size(),
            Directory::Internal(model_directory),
            position_in_nn,
        )) as Box<dyn AllocatableLayer + Send>;
        new_layer.copy_on_filesystem(self.layer_path.path());
        new_layer
    }

    fn copy_on_filesystem(
        &self,
        layer_path: String,
    ) {
        // Copy the layer to the new directory
        let new_layer_path = self.layer_path.clone();
        let original_path = layer_path;
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
    weights: Option<WrappedMatrix<Weight>>, // Weight matrix (output_size x input_size)
    biases: Option<Vec<Bias>>,              // Bias vector (output_size)
    input_cache: Option<Vec<f64>>,          // Cache input for use in backward pass
    input_batch_cache: Option<Vec<Vec<f64>>>, // Cache batch input for use in backward pass
    in_use: bool,
    layer_path: Directory,
}

impl TrainableDenseLayer {
    /// Creates a new `TrainableDenseLayer` with given input and output sizes.
    #[must_use]
    pub fn new(
        input_size: usize,
        output_size: usize,
        model_directory: Directory,
        position_in_nn: usize,
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
        Self {
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
    fn initialize_weights(&self) {
        let mut rng = rand::thread_rng();
        // initialize weights from -0.5 to 0.5
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                let value = rng.gen_range(-0.5..0.5);
                let w = Weight { value, grad: 0.0, m: 0.0, v: 0.0 };
                self.weights.as_ref().unwrap().set_mut_unchecked(i, j, w);
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
        if self.layer_path.exists() {
            // if the layer_path exists, read the matrix and store it
            let (weights, biases) = read_weight(self.layer_path.path())
                .expect("Failed to read layer weights and biases");
            if self.rows == weights.rows() && self.cols == weights.cols() {
                self.rows = weights.rows();
                self.cols = weights.cols();
                self.weights = Some(weights);
                self.biases = Some(biases);
            } else {
                self.weights = Some(WrappedMatrix::new(self.rows, self.cols));
                self.biases = Some(vec![Bias::default(); self.rows]);
                self.initialize_weights();
                save_weight(
                    self.layer_path.path(),
                    self.weights.as_ref().unwrap(),
                    self.biases.as_ref().unwrap(),
                )
                .expect("Failed to save layer weights and biases");
            }
        } else {
            self.weights = Some(WrappedMatrix::new(self.rows, self.cols));
            self.biases = Some(vec![Bias::default(); self.rows]);
            self.initialize_weights();
            save_weight(
                self.layer_path.path(),
                self.weights.as_ref().unwrap(),
                self.biases.as_ref().unwrap(),
            )
            .expect("Failed to save layer weights and biases");
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
    fn forward(
        &mut self,
        input: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        assert!(self.is_allocated(), "Layer not allocated");
        self.input_cache = Some(input.to_vec()); // Cache the input for backpropagation
        let weights = self.weights.as_ref().unwrap().clone();
        let biases = self.biases.as_ref().unwrap().clone();
        let biases_values: Vec<f64> = biases.iter().map(|b| b.value).collect();
        let inputs = input.to_vec();
        utils.execute(move || weights.forward(&inputs, &biases_values))
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
        path: String,
    ) -> Result<(), Box<dyn Error>> {
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
        let weights = WrappedMatrix::new(
            self.weights.as_ref().unwrap().rows(),
            self.weights.as_ref().unwrap().cols(),
        );
        let mut biases = vec![0.0; self.biases.as_ref().unwrap().len()];
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                weights.set_mut_unchecked(
                    i,
                    j,
                    self.weights.as_ref().unwrap().get_unchecked(i, j).value,
                );
            }
        }
        for (i, bias) in self.biases.as_ref().unwrap().iter().enumerate() {
            biases[i] = bias.value;
        }
        save(path, &weights, &biases)
    }

    fn read(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read(path)?;
        self.rows = weights.rows();
        self.cols = weights.cols();
        self.allocate();
        // assign all weights and biases
        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    let w = Weight { value: v, grad: 0.0, m: 0.0, v: 0.0 };
                    self.weights.as_mut().unwrap().set_mut_unchecked(i, j, w);
                }
            }
            if i < biases.len() {
                self.biases.as_mut().unwrap()[i].value = biases[i];
            }
        }
        Ok(())
    }

    fn get_weights(&self) -> WrappedMatrix<f64> {
        let weights = WrappedMatrix::new(
            self.weights.as_ref().unwrap().rows(),
            self.weights.as_ref().unwrap().cols(),
        );
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                let v = self.weights.as_ref().unwrap().get_unchecked(i, j).value;
                weights.set_mut_unchecked(i, j, v);
            }
        }
        weights
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.as_ref().unwrap().iter().map(|bias| bias.value).collect()
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
    fn backward(
        &mut self,
        d_out: &[f64],
        utils: WrappedUtils,
    ) -> Vec<f64> {
        let weights = self.weights.as_ref().unwrap().clone();
        let input_cache = self.input_cache.as_ref().unwrap().clone();
        let d_out_vec = d_out.to_vec();
        // Calculate weight gradients
        let _ = utils.execute(move || {
            weights.backward_calculate_gradients(&d_out_vec, &input_cache);
            0
        });

        let weights_sec = self.weights.as_ref().unwrap().clone();
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
        assert!(self.is_allocated(), "Layer not allocated");
        let weights = self.weights.as_ref().unwrap().clone();
        // Update weight
        let _ = utils.execute(move || {
            weights.update_weights(learning_rate);
            0
        });

        // Update biases
        for bias in self.biases.as_mut().unwrap().iter_mut() {
            bias.value -= learning_rate * bias.grad;
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn backward_batch(
        &mut self,
        _grad_output: &[f64],
    ) -> Vec<f64> {
        unimplemented!()
    }

    fn assign_weights(
        &mut self,
        other: WrappedTrainableLayer,
    ) {
        let weights = other.get_weights();

        let biases = other.get_biases();

        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                if i < weights.rows() && j < weights.cols() {
                    let v = weights.get_unchecked(i, j);
                    let w = Weight { value: v, grad: 0.0, m: 0.0, v: 0.0 };
                    self.weights.as_mut().unwrap().set_mut_unchecked(i, j, w);
                }
            }
            if i < biases.len() {
                self.biases.as_mut().unwrap()[i].value = biases[i];
            }
        }
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
        let weights = self.weights.as_ref().unwrap().clone();
        // Update weights
        let _ = utils.execute(move || {
            weights.adjust_adam(beta1, beta2, epsilon, t, learning_rate);
            0
        });

        // Update biases
        for i in 0..self.biases.as_ref().unwrap().len() {
            let grad = self.biases.as_ref().unwrap()[i].grad;

            // Update first and second moments
            self.biases.as_mut().unwrap()[i].m =
                beta1.mul_add(self.biases.as_ref().unwrap()[i].m, (1.0 - beta1) * grad);
            self.biases.as_mut().unwrap()[i].v =
                beta2.mul_add(self.biases.as_ref().unwrap()[i].v, (1.0 - beta2) * grad.powi(2));

            // Bias correction
            let t_i: i32 = NumCast::from(t).expect("Failed to convert t to i32");
            let m_hat = self.biases.as_ref().unwrap()[i].m / (1.0 - beta1.powi(t_i));
            let v_hat = self.biases.as_ref().unwrap()[i].v / (1.0 - beta2.powi(t_i));

            // Adjusted learning rate
            let adjusted_learning_rate = learning_rate / (v_hat.sqrt() + epsilon);

            // Update biases
            self.biases.as_mut().unwrap()[i].value -= adjusted_learning_rate * m_hat;
        }
    }

    fn save_weight(
        &self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
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
        let weights = WrappedMatrix::new(
            self.weights.as_ref().unwrap().rows(),
            self.weights.as_ref().unwrap().cols(),
        );
        let mut biases = vec![Bias::default(); self.biases.as_ref().unwrap().len()];
        for i in 0..self.weights.as_ref().unwrap().rows() {
            for j in 0..self.weights.as_ref().unwrap().cols() {
                weights.set_mut_unchecked(i, j, self.weights.as_ref().unwrap().get_unchecked(i, j));
            }
        }
        for (i, bias) in self.biases.as_ref().unwrap().iter().enumerate() {
            biases[i] = *bias;
        }
        save_weight(path, &weights, &biases)
    }

    fn read_weight(
        &mut self,
        path: String,
    ) -> Result<(), Box<dyn Error>> {
        // Read weights and biases from a file at the specified path
        let (weights, biases) = read_weight(path)?;
        self.rows = weights.rows();
        self.cols = weights.cols();
        self.allocate();
        // assign all weights and biases
        for i in 0..weights.rows() {
            for j in 0..weights.cols() {
                if i < weights.rows() && j < weights.cols() {
                    self.weights.as_mut().unwrap().set_mut_unchecked(
                        i,
                        j,
                        weights.get_unchecked(i, j),
                    );
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
        let new_layer = Box::new(Self::new(
            self.input_size(),
            self.output_size(),
            Directory::Internal(model_directory),
            position_in_nn,
        )) as Box<dyn TrainableAllocatableLayer + Send>;
        new_layer.copy_on_filesystem(self.layer_path.path());
        new_layer
    }

    fn copy_on_filesystem(
        &self,
        layer_path: String,
    ) {
        // Copy the layer to the new directory
        let new_layer_path = self.layer_path.clone();
        let original_path = layer_path;
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

fn save(
    path: String,
    weights: &WrappedMatrix<f64>,
    biases: &[f64],
) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    let p = Path::new(&path);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).expect("Failed to create directory");
    }
    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
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
    for bias in biases {
        write!(file, "{bias}; ")?;
    }
    writeln!(file)?;
    Ok(())
}

fn save_weight(
    path: String,
    weights: &WrappedMatrix<Weight>,
    biases: &[Bias],
) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    let p = Path::new(&path);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).expect("Failed to create directory");
    }

    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    // Save weights and biases to a file at the specified path
    let mut file = File::create(path)?;
    writeln!(file, "{} {}", weights.rows(), weights.cols())?;
    for i in 0..weights.rows() {
        for j in 0..weights.cols() {
            let weight = weights.get_unchecked(i, j);
            write!(file, "{} {} {} {};", weight.value, weight.grad, weight.m, weight.v)?;
        }
        writeln!(file)?;
    }
    for bias in biases {
        write!(file, "{} {} {} {};", bias.value, bias.grad, bias.m, bias.v)?;
    }
    writeln!(file)?;
    Ok(())
}

fn read(path: String) -> Result<(WrappedMatrix<f64>, Vec<f64>), Box<dyn Error>> {
    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut weights = WrappedMatrix::new(1, 1);
    let mut biases = vec![0.0; 1];
    if let Some(Ok(line)) = lines.next() {
        let mut parts = line.split_whitespace();
        let rows = parts.next().unwrap().parse::<usize>()?;
        let cols = parts.next().unwrap().parse::<usize>()?;
        weights = WrappedMatrix::new(rows, cols);
        for i in 0..rows {
            if let Some(Ok(line)) = lines.next() {
                let parts = line.split(';').collect::<Vec<_>>();
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
                            weights.set_mut_unchecked(i, j, figures[0].parse::<f64>()?);
                        } else {
                            return Err("Invalid weight format".into());
                        }
                    }
                }
            }
        }
    }
    if let Some(Ok(line)) = lines.next() {
        let parts = line.split(';').collect::<Vec<_>>();
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

fn read_weight(path: String) -> Result<(WrappedMatrix<Weight>, Vec<Bias>), Box<dyn Error>> {
    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut weights = WrappedMatrix::new(1, 1);
    let mut biases = vec![Bias::default(); 1];
    if let Some(Ok(line)) = lines.next() {
        let mut dim_parts = line.split_whitespace();
        let rows = dim_parts.next().unwrap().parse::<usize>()?;
        let cols = dim_parts.next().unwrap().parse::<usize>()?;
        weights = WrappedMatrix::new(rows, cols);
        for i in 0..rows {
            if let Some(Ok(line)) = lines.next() {
                let parts = line.split(';').collect::<Vec<_>>();
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
                            weights.set_mut_unchecked(
                                i,
                                j,
                                Weight {
                                    value: figures[0].parse::<f64>()?,
                                    grad: figures[1].parse::<f64>()?,
                                    m: figures[2].parse::<f64>()?,
                                    v: figures[3].parse::<f64>()?,
                                },
                            );
                        } else if figures.len() == 1 {
                            weights.set_mut_unchecked(
                                i,
                                j,
                                Weight {
                                    value: figures[0].parse::<f64>()?,
                                    grad: 0.0,
                                    m: 0.0,
                                    v: 0.0,
                                },
                            );
                        } else {
                            return Err("Invalid weight format".into());
                        }
                    }
                }
            }
        }
    }
    if let Some(Ok(line)) = lines.next() {
        let parts = line.split(';').collect::<Vec<_>>();
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
    use crate::utilities::util::Utils;

    use super::*;

    #[test]
    fn test_dense_layer() {
        let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
        let mut layer =
            TrainableDenseLayer::new(3, 2, Directory::Internal("test_model_unit".to_string()), 0);

        let input = vec![1.0, 2.0, 3.0];
        layer.allocate();
        layer.mark_for_use();
        let output = layer.forward(&input, utils.clone());
        layer.free_from_use();

        assert_eq!(output.len(), 2);

        let grad_output = vec![0.1, 0.2];
        let grad_input = layer.backward(&grad_output, utils.clone());

        assert_eq!(grad_input.len(), 3);

        layer.update_weights(0.01, utils);

        std::fs::remove_dir_all("test_model_unit").unwrap();
    }
}

#[derive(Debug, Default, Clone)]
pub struct NumberEntry(f64);

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
pub struct WeightEntry(Weight);

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
