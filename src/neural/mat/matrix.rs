use std::fmt;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

#[derive(Debug)]
pub struct OutOfRangeError {
    message: String,
}

impl fmt::Display for OutOfRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for OutOfRangeError {}

impl<T> Matrix<T>
where
    T: Default + Clone,
{
    // Constructor with specified rows and columns
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    // Default constructor (1x1 matrix)
    pub fn default() -> Self {
        Self::new(1, 1)
    }

    // Get a mutable reference to an element at (x, y)
    pub fn get_mut(&mut self, x: usize, y: usize) -> Result<&mut T, OutOfRangeError> {
        if x >= self.rows || y >= self.cols {
            Err(OutOfRangeError {
                message: format!(
                    "Matrix::get_mut out of range x: {}, y: {}, rows: {}, cols: {}",
                    x, y, self.rows, self.cols
                ),
            })
        } else {
            Ok(&mut self.data[x * self.cols + y])
        }
    }

    // Get an immutable reference to an element at (x, y)
    pub fn get(&self, x: usize, y: usize) -> Result<&T, OutOfRangeError> {
        if x >= self.rows || y >= self.cols {
            Err(OutOfRangeError {
                message: format!(
                    "Matrix::get out of range x: {}, y: {}, rows: {}, cols: {}",
                    x, y, self.rows, self.cols
                ),
            })
        } else {
            Ok(&self.data[x * self.cols + y])
        }
    }

    pub fn get_mut_unchecked(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[x * self.cols + y]
    }

    pub fn get_unchecked(&self, x: usize, y: usize) -> &T {
        &self.data[x * self.cols + y]
    }

    // Return the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    // Return the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }
}