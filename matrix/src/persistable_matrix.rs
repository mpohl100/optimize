use crate::directory::Directory;
use crate::mat::WrappedMatrix;
use alloc::allocatable::Allocatable;
use fs2::FileExt;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;

pub trait PersistableValue: Default + Clone {
    fn to_string_for_matrix(&self) -> String;
    /// # Errors
    /// Returns an error if the string cannot be parsed into the type.
    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}

pub struct PersistableMatrix<T: PersistableValue> {
    matrix_file_path: Directory,
    rows: usize,
    cols: usize,
    mat: Option<WrappedMatrix<T>>,
    in_use: bool,
}

impl<T: PersistableValue> PersistableMatrix<T> {
    #[must_use]
    pub fn new(
        matrix_file_path: Directory,
        label: &str,
        rows: usize,
        cols: usize,
    ) -> Self {
        let matrix_path = match matrix_file_path {
            Directory::User(path) => Directory::User(format!("{path}/matrix_{label}.txt")),
            Directory::Internal(path) => Directory::Internal(format!("{path}/matrix_{label}.txt")),
        };

        Self { matrix_file_path: matrix_path, rows, cols, mat: None, in_use: false }
    }
}

impl<T: PersistableValue> Drop for PersistableMatrix<T> {
    fn drop(&mut self) {
        // Save the model to ensure that everything is on disk if it is a user_model_directory
        if let Directory::User(dir) = &self.matrix_file_path {
            if std::fs::metadata(dir).is_ok() {
                // Save the model to disk
                self.deallocate();
            }
        }
        // Remove the internal model directory from disk
        if let Directory::Internal(dir) = &self.matrix_file_path {
            // check that dir is a file
            let path = Path::new(dir);
            // delete the file
            if path.is_file() {
                std::fs::remove_file(dir).expect("Failed to remove file");
            }
        }
    }
}

impl<T: PersistableValue> Allocatable for PersistableMatrix<T> {
    fn allocate(&mut self) {
        if self.is_allocated() {
            return;
        }
        // if the layer_path does not exist, create a new matrix and store it
        if self.matrix_file_path.exists() {
            // if the layer_path exists, read the matrix and store it
            let matrix = read(self.matrix_file_path.path()).expect("Failed to read layer weights");
            if self.rows == matrix.rows() && self.cols == matrix.cols() {
                self.rows = matrix.rows();
                self.cols = matrix.cols();
                self.mat = Some(matrix);
            }
        } else {
            self.mat = Some(WrappedMatrix::new(self.rows, self.cols));
            save(self.matrix_file_path.path(), self.mat.as_ref().unwrap())
                .expect("Failed to save layer weights");
        }
    }

    fn deallocate(&mut self) {
        if self.is_allocated() {
            save(self.matrix_file_path.path(), self.mat.as_ref().unwrap())
                .expect("Failed to save matrix");
        }
        self.mat = None;
    }

    fn is_allocated(&self) -> bool {
        self.mat.is_some()
    }

    fn get_size(&self) -> usize {
        (self.rows * self.cols) * std::mem::size_of::<T>()
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

fn save<T: PersistableValue>(
    path: String,
    matrix: &WrappedMatrix<T>,
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
    writeln!(file, "{} {}", matrix.rows(), matrix.cols())?;
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            write!(file, "{};", matrix.get_unchecked(i, j).to_string_for_matrix())?;
        }
        writeln!(file)?;
    }
    writeln!(file)?;
    Ok(())
}

fn read<T: PersistableValue>(path: String) -> Result<(WrappedMatrix<T>), Box<dyn Error>> {
    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut weights = WrappedMatrix::new(1, 1);
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
                            let val = T::from_string_for_matrix(figures[0])?;
                            weights.set_mut_unchecked(i, j, val);
                        } else {
                            return Err("Invalid weight format".into());
                        }
                    }
                }
            }
        }
    }
    Ok(weights)
}
