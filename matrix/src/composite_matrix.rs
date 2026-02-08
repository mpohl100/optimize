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
    directory: Directory,
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
                    &format!("composite_{i}_{j}"),
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
            directory: directory.clone(),
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

    #[must_use]
    pub fn get_directory(&self) -> Directory {
        self.directory.clone()
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

    /// Set a submatrix within the composite matrix
    /// # Panics
    /// Panics if the submatrix dimensions exceed the bounds of the composite matrix
    pub fn set_submatrix(
        &self,
        start_x: usize,
        start_y: usize,
        submatrix: &Self,
    ) {
        // check that dimensions are not exceeded
        assert!(
            !(start_x + submatrix.rows() > self.rows() || start_y + submatrix.cols() > self.cols()),
            "Submatrix dimensions exceed bounds of composite matrix"
        );
        for x in 0..submatrix.rows() {
            for y in 0..submatrix.cols() {
                let value = submatrix.get_unchecked(x, y);
                self.set_mut_unchecked(start_x + x, start_y + y, value);
            }
        }
    }

    /// Get a submatrix from the composite matrix
    /// # Panics
    /// Panics if the submatrix dimensions exceed the bounds of the composite matrix
    #[must_use]
    pub fn get_submatrix(
        &self,
        start_x: usize,
        start_y: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        // check that dimensions are not exceeded
        assert!(
            !(start_x + rows > self.rows() || start_y + cols > self.cols()),
            "Submatrix dimensions exceed bounds of composite matrix"
        );
        let internal_directory =
            self.get_directory().expand(&format!("submatrix_{start_x}_{start_y}")).to_internal();
        let submatrix = CompositeMatrix::new(
            self.cm.lock().unwrap().get_slice_num_cols(),
            self.cm.lock().unwrap().get_slice_num_rows(),
            rows,
            cols,
            &internal_directory,
            self.get_alloc_manager(),
        );
        let wrapped_submatrix = Self::new(submatrix);
        for x in 0..rows {
            for y in 0..cols {
                let value = self.get_unchecked(start_x + x, start_y + y);
                wrapped_submatrix.set_mut_unchecked(x, y, value);
            }
        }
        wrapped_submatrix
    }

    #[must_use]
    pub fn get_directory(&self) -> Directory {
        let cm = safe_lock(&self.cm);
        cm.get_directory()
    }
}
