use std::error::Error;
use std::fmt;
use std::slice;
use std::sync::Arc;
use std::sync::Mutex;

use rayon::prelude::*;

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
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for OutOfRangeError {}

impl<T> Matrix<T>
where
    T: Default + Clone,
{
    // Constructor with specified rows and columns
    #[must_use] pub fn new(
        rows: usize,
        cols: usize,
    ) -> Self {
        Self { rows, cols, data: vec![T::default(); rows * cols] }
    }

    // Get a mutable reference to an element at (x, y)
    pub fn get_mut(
        &mut self,
        x: usize,
        y: usize,
    ) -> Result<&mut T, OutOfRangeError> {
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
    pub fn get(
        &self,
        x: usize,
        y: usize,
    ) -> Result<&T, OutOfRangeError> {
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

    pub fn get_mut_unchecked(
        &mut self,
        x: usize,
        y: usize,
    ) -> &mut T {
        &mut self.data[x * self.cols + y]
    }

    #[must_use] pub fn get_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> &T {
        &self.data[x * self.cols + y]
    }

    pub fn set_mut_unchecked(
        &mut self,
        x: usize,
        y: usize,
        value: T,
    ) {
        self.data[x * self.cols + y] = value;
    }

    // Return the number of rows
    #[must_use] pub const fn rows(&self) -> usize {
        self.rows
    }

    // Return the number of columns
    #[must_use] pub const fn cols(&self) -> usize {
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
    #[must_use] pub const fn iter(&self) -> RowIter<T> {
        RowIter { matrix: self, current_row: 0 }
    }

    /// Returns an iterator over the rows of the matrix (mutable)
    pub fn iter_mut(&mut self) -> RowIterMut<T> {
        RowIterMut { matrix: self, current_row: 0 }
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
                slice::from_raw_parts_mut(
                    self.matrix.data.as_mut_ptr().add(start),
                    self.matrix.cols,
                )
            })
        } else {
            None
        }
    }
}

impl<T: Sync> Matrix<T> {
    #[must_use] pub fn par_iter(&self) -> rayon::slice::Chunks<'_, T> {
        self.data.par_chunks(self.cols)
    }

    #[must_use] pub fn par_indexed_iter(&self) -> impl ParallelIterator<Item = (usize, &[T])> {
        self.data.par_chunks(self.cols).enumerate()
    }
}

impl<T: Send> Matrix<T> {
    pub fn par_iter_mut(&mut self) -> rayon::slice::ChunksMut<'_, T> {
        self.data.par_chunks_mut(self.cols)
    }

    pub fn par_indexed_iter_mut(&mut self) -> impl ParallelIterator<Item = (usize, &mut [T])> {
        self.data.par_chunks_mut(self.cols).enumerate()
    }
}

#[derive(Debug, Clone)]
pub struct WrappedMatrix<T: Default + Clone> {
    mat: Arc<Mutex<Matrix<T>>>,
}

impl<T> WrappedMatrix<T>
where
    T: Default + Clone,
{
    #[must_use] pub fn new(
        rows: usize,
        cols: usize,
    ) -> Self {
        Self { mat: Arc::new(Mutex::new(Matrix::<T>::new(rows, cols))) }
    }

    pub fn get(
        &self,
        x: usize,
        y: usize,
    ) -> Result<T, OutOfRangeError> {
        let mat = self.mat.lock().unwrap();
        mat.get(x, y).cloned()
    }

    pub fn get_mut(
        &self,
        x: usize,
        y: usize,
    ) -> Result<T, OutOfRangeError> {
        let mut mat = self.mat.lock().unwrap();
        mat.get_mut(x, y).cloned()
    }

    #[must_use] pub fn rows(&self) -> usize {
        let mat = self.mat.lock().unwrap();
        mat.rows()
    }

    #[must_use] pub fn cols(&self) -> usize {
        let mat = self.mat.lock().unwrap();
        mat.cols()
    }

    #[must_use] pub fn mat(&self) -> Arc<Mutex<Matrix<T>>> {
        self.mat.clone()
    }

    pub fn set_mut_unchecked(
        &self,
        x: usize,
        y: usize,
        value: T,
    ) {
        let mut mat = self.mat.lock().unwrap();
        mat.set_mut_unchecked(x, y, value);
    }

    #[must_use] pub fn get_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> T {
        let mat = self.mat.lock().unwrap();
        mat.get_unchecked(x, y).clone()
    }
}
