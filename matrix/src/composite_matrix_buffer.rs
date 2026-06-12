use crate::matrix_buffer::WrappedMatrixBuffer;

use utils::safer::safe_lock;

use std::sync::{Arc, Mutex};

pub struct CompositeMatrixBuffer<T: Default + Clone> {
    tile_x: usize,
    tile_y: usize,
    rows: usize,
    cols: usize,
    buffers: Vec<WrappedMatrixBuffer<T>>,
}

impl<T: Default + Clone> CompositeMatrixBuffer<T> {
    /// Creates a new `CompositeMatrixBuffer` with the specified dimensions and tile sizes.
    #[must_use]
    pub fn new(
        tile_x: usize,
        tile_y: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        let buffers = (0..(rows / tile_x + 1) * (cols / tile_y + 1))
            .map(|_| WrappedMatrixBuffer::new(tile_x, tile_y))
            .collect::<Vec<_>>();
        Self { tile_x, tile_y, rows, cols, buffers }
    }

    /// Gets a value from the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    #[must_use]
    pub fn get_val(
        &self,
        row: usize,
        col: usize,
    ) -> Option<T> {
        if row < self.rows && col < self.cols {
            let buffer_row = row / self.tile_x;
            let buffer_col = col / self.tile_y;
            let buffer_index = buffer_row * (self.cols / self.tile_y + 1) + buffer_col;
            let buffer = self.buffers.get(buffer_index)?;
            let inner_row = row % self.tile_x;
            let inner_col = col % self.tile_y;
            buffer.get_val(inner_row, inner_col)
        } else {
            None
        }
    }

    /// Sets a value in the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `Err` if the provided indices are out of bounds.
    fn set_val(
        &mut self,
        row: usize,
        col: usize,
        value: T,
    ) -> Result<(), String> {
        if row < self.rows && col < self.cols {
            let buffer_row = row / self.tile_x;
            let buffer_col = col / self.tile_y;
            let buffer_index = buffer_row * (self.cols / self.tile_y + 1) + buffer_col;
            let buffer = self.buffers.get_mut(buffer_index).ok_or("Buffer not found")?;
            let inner_row = row % self.tile_x;
            let inner_col = col % self.tile_y;
            buffer.set_val(inner_row, inner_col, value)
        } else {
            Err("Indices out of bounds".to_string())
        }
    }

    #[must_use]
    pub fn get_sub_matrix_buffer(
        &self,
        sub_row: usize,
        sub_col: usize,
    ) -> Option<WrappedMatrixBuffer<T>> {
        let buffer_index = sub_row * (self.cols / self.tile_y) + sub_col;
        self.buffers.get(buffer_index).cloned()
    }

    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[must_use]
    pub const fn sub_matrix_shape(&self) -> (usize, usize) {
        (self.tile_x, self.tile_y)
    }

    #[must_use]
    pub const fn num_sub_matrices(&self) -> (usize, usize) {
        (self.rows / self.tile_x, self.cols / self.tile_y)
    }
}

#[derive(Clone)]
pub struct WrappedCompositeMatrixBuffer<T: Default + Clone> {
    buffer: Arc<Mutex<CompositeMatrixBuffer<T>>>,
}

impl<T: Default + Clone> WrappedCompositeMatrixBuffer<T> {
    #[must_use]
    pub fn new(
        tile_x: usize,
        tile_y: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(CompositeMatrixBuffer::new(tile_x, tile_y, rows, cols))),
        }
    }

    /// Gets a value from the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    /// # Panics
    /// Panics if the internal locking mechanism fails (e.g., if the mutex is poisoned
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        let buffer = self.buffer.lock().unwrap();
        buffer.shape()
    }

    /// Gets a value from the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    /// # Panics
    /// Panics if the internal locking mechanism fails (e.g., if the mutex is poisoned
    #[must_use]
    pub fn sub_matrix_shape(&self) -> (usize, usize) {
        let buffer = safe_lock(&self.buffer);
        buffer.sub_matrix_shape()
    }

    /// Gets a value from the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    /// # Panics
    /// Panics if the internal locking mechanism fails (e.g., if the mutex is poisoned
    #[must_use]
    pub fn num_sub_matrices(&self) -> (usize, usize) {
        let buffer = safe_lock(&self.buffer);
        buffer.num_sub_matrices()
    }

    /// Gets a value from the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `None` if the provided indices are out of bounds.
    /// # Panics
    /// Panics if the internal locking mechanism fails (e.g., if the mutex is poisoned
    #[must_use]
    pub fn get_val(
        &self,
        row: usize,
        col: usize,
    ) -> Option<T> {
        let buffer = safe_lock(&self.buffer);
        buffer.get_val(row, col)
    }

    /// Sets a value in the composite matrix buffer at the specified row and column.
    /// # Errors
    /// Returns `Err` if the provided indices are out of bounds.
    /// # Panics
    /// Panics if the internal locking mechanism fails (e.g., if the mutex is poisoned
    pub fn set_val(
        &self,
        row: usize,
        col: usize,
        value: T,
    ) -> Result<(), String> {
        let mut buffer = safe_lock(&self.buffer);
        buffer.set_val(row, col, value)
    }

    /// Gets a sub-matrix buffer for the specified sub-matrix indices.
    /// # Arguments
    /// * `sub_row` - The row index of the sub-matrix.
    /// * `sub_col` - The column index of the sub-matrix.
    /// # Returns
    /// An `Option` containing the `WrappedMatrixBuffer` for the specified sub-matrix if the indices are valid, or `None` if the indices are out of bounds. The sub-matrix buffer allows access to the individual elements of the sub-matrix corresponding to the given indices.
    /// # Errors
    /// Returns `None` if the provided sub-matrix indices are out of bounds.
    /// # Panics
    /// Panics if the internal locking mechanism fails (e.g., if the mutex is poisoned
    #[must_use]
    pub fn get_sub_matrix_buffer(
        &self,
        sub_row: usize,
        sub_col: usize,
    ) -> Option<WrappedMatrixBuffer<T>> {
        let buffer = safe_lock(&self.buffer);
        buffer.get_sub_matrix_buffer(sub_row, sub_col)
    }
}
