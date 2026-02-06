use crate::directory::Directory;
use crate::mat::WrappedMatrix;
use crate::persistable_matrix::{PersistableMatrix, PersistableValue, WrappedPersistableMatrix};

use alloc::alloc_manager::WrappedAllocManager;
use utils::safer::safe_lock;

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct CompositeMatrix<T: PersistableValue + From<f64> + 'static> {
    slice_num_cols: usize,
    slice_num_rows: usize,
    rows: usize,
    cols: usize,
    matrices: WrappedMatrix<WrappedPersistableMatrix<T>>,
    wrapped_alloc_manager: WrappedAllocManager<WrappedPersistableMatrix<T>>,
}

impl<T: PersistableValue + From<f64> + 'static> CompositeMatrix<T> {
    ///  Create a new ``CompositeMatrix``
    /// # Panics
    /// Panics if ``set_mut_unchecked`` fails
    #[must_use]
    pub fn new(
        slice_x: usize,
        slice_y: usize,
        rows: usize,
        cols: usize,
        directory: &Directory,
        wrapped_alloc_manager: WrappedAllocManager<WrappedPersistableMatrix<T>>,
    ) -> Self {
        let mat_rows = rows / slice_x + 1;
        let mat_cols = cols / slice_y + 1;
        let matrices = WrappedMatrix::new(mat_rows, mat_cols);
        for i in 0..mat_rows {
            for j in 0..mat_cols {
                let persistable_rows = if i == mat_rows - 1 { rows % slice_x } else { slice_x };
                let persistable_cols = if j == mat_cols - 1 { cols % slice_y } else { slice_y };
                let persistable_matrix = WrappedPersistableMatrix::new(PersistableMatrix::new(
                    directory.clone(),
                    &format!("composite_{}_{}_{}", i, j, std::any::type_name::<T>()),
                    persistable_rows,
                    persistable_cols,
                ));
                matrices.mat().lock().unwrap().set_mut_unchecked(i, j, persistable_matrix);
            }
        }
        Self {
            slice_num_cols: slice_x,
            slice_num_rows: slice_y,
            rows,
            cols,
            matrices,
            wrapped_alloc_manager,
        }
    }

    /// Set the value at (x, y) without bounds checking.
    /// # Panics
    /// Panics if ``set_mut_unchecked`` fails
    pub fn set_mut_unchecked(
        &self,
        x: usize,
        y: usize,
        value: T,
    ) {
        let matrix_x = x / self.slice_num_cols;
        let matrix_y = y / self.slice_num_rows;
        let within_x = x % self.slice_num_cols;
        let within_y = y % self.slice_num_rows;
        let persistable_matrix = self.matrices.get_mut_unchecked(matrix_x, matrix_y);
        persistable_matrix.set_mut_unchecked(within_x, within_y, value);
    }

    #[must_use]
    pub const fn matrices(&self) -> &WrappedMatrix<WrappedPersistableMatrix<T>> {
        &self.matrices
    }

    #[must_use]
    pub const fn get_slice_num_cols(&self) -> usize {
        self.slice_num_cols
    }

    #[must_use]
    pub const fn get_slice_num_rows(&self) -> usize {
        self.slice_num_rows
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Save the composite matrix to disk
    /// # Errors
    /// Returns an error if saving fails
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..(self.rows / self.slice_num_cols) {
            for j in 0..(self.cols / self.slice_num_rows) {
                let persistable_matrix = self.matrices.get_unchecked(i, j);
                persistable_matrix.save()?;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn get_alloc_manager(&self) -> WrappedAllocManager<WrappedPersistableMatrix<T>> {
        self.wrapped_alloc_manager.clone()
    }
}

#[derive(Debug, Clone)]
pub struct WrappedCompositeMatrix<T: PersistableValue + From<f64> + 'static> {
    cm: std::sync::Arc<std::sync::Mutex<CompositeMatrix<T>>>,
}

impl<T: PersistableValue + From<f64> + 'static> WrappedCompositeMatrix<T> {
    #[must_use]
    pub fn new(cm: CompositeMatrix<T>) -> Self {
        Self { cm: std::sync::Arc::new(std::sync::Mutex::new(cm)) }
    }

    #[must_use]
    pub fn get_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> T {
        let cm = safe_lock(&self.cm);
        let matrix_x = x / cm.get_slice_num_cols();
        let matrix_y = y / cm.get_slice_num_rows();
        let within_x = x % cm.get_slice_num_cols();
        let within_y = y % cm.get_slice_num_rows();
        let persistable_matrix = cm.matrices().get_unchecked(matrix_x, matrix_y);
        drop(cm);
        self.get_alloc_manager().allocate(&persistable_matrix);
        persistable_matrix.get_unchecked(within_x, within_y)
    }

    pub fn set_mut_unchecked(
        &self,
        x: usize,
        y: usize,
        value: T,
    ) {
        let cm = safe_lock(&self.cm);
        let matrix_x = x / cm.get_slice_num_cols();
        let matrix_y = y / cm.get_slice_num_rows();
        let within_x = x % cm.get_slice_num_cols();
        let within_y = y % cm.get_slice_num_rows();
        let persistable_matrix = cm.matrices().get_unchecked(matrix_x, matrix_y);
        drop(cm);
        self.get_alloc_manager().allocate(&persistable_matrix);
        persistable_matrix.set_mut_unchecked(within_x, within_y, value);
    }

    /// Save the composite matrix to disk
    /// # Errors
    /// Returns an error if saving fails
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let cm = safe_lock(&self.cm);
        cm.save()
    }

    #[must_use]
    pub fn mat(&self) -> Arc<Mutex<CompositeMatrix<T>>> {
        self.cm.clone()
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        let cm = safe_lock(&self.cm);
        cm.rows()
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        let cm = safe_lock(&self.cm);
        cm.cols()
    }

    /// Get the underlying allocation manager
    #[must_use]
    pub fn get_alloc_manager(&self) -> WrappedAllocManager<WrappedPersistableMatrix<T>> {
        let cm = safe_lock(&self.cm);
        cm.get_alloc_manager()
    }
}
