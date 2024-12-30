use std::error::Error;
use std::fmt;
use std::slice;

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

/// Immutable row iterator
pub struct RowIter<'a, T> {
    matrix: &'a Matrix<T>,
    current_row: usize,
}

/// Mutable row iterator
pub struct RowIterMut<'a, T> {
    matrix: &'a mut Matrix<T>,
    current_row: usize,
}

impl<T> Matrix<T> {
    /// Returns an iterator over the rows of the matrix (immutable)
    pub fn iter(&self) -> RowIter<T> {
        RowIter {
            matrix: self,
            current_row: 0,
        }
    }

    /// Returns an iterator over the rows of the matrix (mutable)
    pub fn iter_mut(&mut self) -> RowIterMut<T> {
        RowIterMut {
            matrix: self,
            current_row: 0,
        }
    }
}

// Implement `Iterator` for `RowIter`
impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.rows {
            let start = self.current_row * self.matrix.cols;
            let end = start + self.matrix.cols;
            self.current_row += 1;
            Some(&self.matrix.data[start..end])
        } else {
            None
        }
    }
}

// Implement `Iterator` for `RowIterMut`
impl<'a, T> Iterator for RowIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row < self.matrix.rows {
            let start = self.current_row * self.matrix.cols;
            let _end = start + self.matrix.cols;
            self.current_row += 1;
            // SAFETY: This is safe because we ensure each slice is disjoint
            Some(unsafe {
                slice::from_raw_parts_mut(self.matrix.data.as_mut_ptr().add(start), self.matrix.cols)
            })
        } else {
            None
        }
    }
}

