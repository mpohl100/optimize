use utils::safer::safe_lock;

use std::sync::{Arc, Mutex};

struct MatrixBuffer<T: Default + Clone> {
    buffer: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Default + Clone> MatrixBuffer<T> {
    /// Creates a new `MatrixBuffer` with the specified number of rows and columns.
    #[must_use]
    fn new(
        rows: usize,
        cols: usize,
    ) -> Self {
        let buffer = vec![T::default(); rows * cols];
        Self { buffer, rows, cols }
    }

    /// Gets a value from the matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    #[must_use]
    fn get(
        &self,
        row: usize,
        col: usize,
    ) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.buffer[row * self.cols + col])
        } else {
            None
        }
    }

    /// Sets a value in the matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `Err` if the provided indices are out of bounds.
    fn set(
        &mut self,
        row: usize,
        col: usize,
        value: T,
    ) -> Result<(), String> {
        if row < self.rows && col < self.cols {
            self.buffer[row * self.cols + col] = value;
            Ok(())
        } else {
            Err("Index out of bounds".to_string())
        }
    }
}

pub struct WrappedMatrixBuffer<T: Default + Clone> {
    buffer: Arc<Mutex<MatrixBuffer<T>>>,
}

impl<T: Default + Clone> WrappedMatrixBuffer<T> {
    /// Creates a new `WrappedMatrixBuffer` with the specified number of rows and columns.
    #[must_use]
    pub fn new(
        rows: usize,
        cols: usize,
    ) -> Self {
        let buffer = MatrixBuffer::new(rows, cols);
        Self { buffer: Arc::new(Mutex::new(buffer)) }
    }

    /// Gets a value from the matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    #[must_use]
    pub fn get(
        &self,
        row: usize,
        col: usize,
    ) -> Option<T> {
        let buffer = safe_lock(&self.buffer);
        buffer.get(row, col).cloned()
    }

    /// Sets a value in the matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `Err` if the provided indices are out of bounds.
    pub fn set(
        &self,
        row: usize,
        col: usize,
        value: T,
    ) -> Result<(), String> {
        let mut buffer = safe_lock(&self.buffer);
        buffer.set(row, col, value)
    }
}
