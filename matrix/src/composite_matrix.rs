use crate::directory::Directory;
use crate::mat::WrappedMatrix;
use crate::persistable_matrix::{PersistableMatrix, PersistableValue};

use utils::safer::safe_lock;

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct CompositeMatrix<T: PersistableValue> {
    slice_x: usize,
    slice_y: usize,
    rows: usize,
    cols: usize,
    matrices: WrappedMatrix<PersistableMatrix<T>>,
}

impl<T: PersistableValue> CompositeMatrix<T> {
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
    ) -> Self {
        let matrices = WrappedMatrix::new(rows / slice_x + 1, cols / slice_y + 1);
        for i in 0..=(rows / slice_x) {
            for j in 0..=(cols / slice_y) {
                let persistable_matrix = PersistableMatrix::new(
                    directory.clone(),
                    &format!("composite_{}_{}_{}", i, j, std::any::type_name::<T>()),
                    slice_x,
                    slice_y,
                );
                matrices.mat().lock().unwrap().set_mut_unchecked(i, j, persistable_matrix);
            }
        }
        Self { slice_x, slice_y, rows, cols, matrices }
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
        let matrix_x = x / self.slice_x;
        let matrix_y = y / self.slice_y;
        let within_x = x % self.slice_x;
        let within_y = y % self.slice_y;
        let mut persistable_matrix = self.matrices.get_mut_unchecked(matrix_x, matrix_y);
        persistable_matrix.set_mut_unchecked(within_x, within_y, value);
    }

    #[must_use]
    pub const fn matrices(&self) -> &WrappedMatrix<PersistableMatrix<T>> {
        &self.matrices
    }

    #[must_use]
    pub const fn get_slice_x(&self) -> usize {
        self.slice_x
    }

    #[must_use]
    pub const fn get_slice_y(&self) -> usize {
        self.slice_y
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
        for i in 0..(self.rows / self.slice_x) {
            for j in 0..(self.cols / self.slice_y) {
                let persistable_matrix = self.matrices.get_unchecked(i, j);
                persistable_matrix.save()?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WrappedCompositeMatrix<T: PersistableValue> {
    cm: std::sync::Arc<std::sync::Mutex<CompositeMatrix<T>>>,
}

impl<T: PersistableValue> WrappedCompositeMatrix<T> {
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
        let matrix_x = x / cm.get_slice_x();
        let matrix_y = y / cm.get_slice_y();
        let within_x = x % cm.get_slice_x();
        let within_y = y % cm.get_slice_y();
        let persistable_matrix = cm.matrices().get_unchecked(matrix_x, matrix_y);
        drop(cm);
        persistable_matrix.get_unchecked(within_x, within_y)
    }

    pub fn set_mut_unchecked(
        &self,
        x: usize,
        y: usize,
        value: T,
    ) {
        let cm = safe_lock(&self.cm);
        cm.set_mut_unchecked(x, y, value);
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
}
