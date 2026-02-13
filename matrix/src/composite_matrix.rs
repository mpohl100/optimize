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

    /// Get the value at (x, y) without bounds checking.
    #[must_use]
    pub fn get_mut_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> T {
        let matrix_x = x / self.slice_num_cols;
        let matrix_y = y / self.slice_num_rows;
        let within_x = x % self.slice_num_cols;
        let within_y = y % self.slice_num_rows;
        let persistable_matrix = self.matrices.get_unchecked(matrix_x, matrix_y);
        self.get_alloc_manager().allocate(&persistable_matrix);
        persistable_matrix.get_unchecked(within_x, within_y)
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
        let persistable_matrix = self.matrices.get_unchecked(matrix_x, matrix_y);
        self.get_alloc_manager().allocate(&persistable_matrix);
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
    ///
    /// # Implementation Note
    /// This method iterates over all sub-matrices (including edge matrices with remainder dimensions).
    /// Previously, this used `self.rows / self.slice_num_cols` which only iterated over complete slices,
    /// missing the remainder matrices at the edges. For example, with a 10x10 matrix divided into 3x3 slices:
    /// - Old loop: 0..(10/3) = 0..3, missing the 4th row/column of matrices
    /// - New loop: ``0..self.matrices.rows()`` = 0..4, correctly saving all 16 sub-matrices
    ///
    /// This bug caused edge weights to appear different because they were never saved/loaded from disk.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..self.matrices.rows() {
            for j in 0..self.matrices.cols() {
                let mut persistable_matrix = self.matrices.get_unchecked(i, j);
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
            !(start_x + submatrix.rows > self.rows || start_y + submatrix.cols > self.cols),
            "Submatrix dimensions exceed bounds of composite matrix"
        );
        for x in 0..submatrix.rows {
            for y in 0..submatrix.cols {
                let matrix_x = x / submatrix.slice_num_cols;
                let matrix_y = y / submatrix.slice_num_rows;
                let within_x = x % submatrix.slice_num_cols;
                let within_y = y % submatrix.slice_num_rows;
                let persistable_matrix = submatrix.matrices.get_unchecked(matrix_x, matrix_y);
                let value = persistable_matrix.get_unchecked(within_x, within_y);
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
        wrapped_alloc_manager: WrappedAllocManager<WrappedPersistableMatrix<T>>,
    ) -> Self {
        // check that dimensions are not exceeded
        assert!(
            !(start_x + rows > self.rows || start_y + cols > self.cols),
            "Submatrix dimensions exceed bounds of composite matrix"
        );
        let internal_directory =
            self.directory.expand(&format!("submatrix_{start_x}_{start_y}")).to_internal();
        let submatrix = Self::new(
            self.slice_num_cols,
            self.slice_num_rows,
            rows,
            cols,
            &internal_directory,
            wrapped_alloc_manager,
        );
        for x in 0..rows {
            for y in 0..cols {
                let matrix_x = (start_x + x) / self.slice_num_cols;
                let matrix_y = (start_y + y) / self.slice_num_rows;
                let within_x = (start_x + x) % self.slice_num_cols;
                let within_y = (start_y + y) % self.slice_num_rows;
                let persistable_matrix = self.matrices.get_unchecked(matrix_x, matrix_y);
                let value = persistable_matrix.get_unchecked(within_x, within_y);
                submatrix.set_mut_unchecked(x, y, value);
            }
        }
        submatrix
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
        let persistable_matrix = cm.matrices().get_unchecked(matrix_x, matrix_y);
        drop(cm);
        self.get_alloc_manager().allocate(&persistable_matrix);
        let cm = safe_lock(&self.cm);
        cm.get_mut_unchecked(x, y)
    }

    #[must_use]
    pub fn get_mut_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> T {
        let cm = safe_lock(&self.cm);
        let matrix_x = x / cm.get_slice_num_cols();
        let matrix_y = y / cm.get_slice_num_rows();
        let persistable_matrix = cm.matrices().get_unchecked(matrix_x, matrix_y);
        drop(cm);
        self.get_alloc_manager().allocate(&persistable_matrix);
        let cm = safe_lock(&self.cm);
        cm.get_mut_unchecked(x, y)
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
        let persistable_matrix = cm.matrices().get_unchecked(matrix_x, matrix_y);
        drop(cm);
        self.get_alloc_manager().allocate(&persistable_matrix);
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
        let cm = safe_lock(&self.cm);
        let sub_cm = safe_lock(&submatrix.cm);
        cm.set_submatrix(start_x, start_y, &sub_cm);
        drop(cm);
        drop(sub_cm);
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
        let cm = safe_lock(&self.cm);
        let alloc_manager = cm.get_alloc_manager();
        let submatrix = cm.get_submatrix(start_x, start_y, rows, cols, alloc_manager);
        drop(cm);
        Self::new(submatrix)
    }

    #[must_use]
    pub fn get_directory(&self) -> Directory {
        let cm = safe_lock(&self.cm);
        let dir = cm.get_directory();
        drop(cm);
        dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_types::NumberEntry;
    use crate::directory::Directory;
    use alloc::alloc_manager::AllocManager;
    use alloc::memory_counter::{MemoryCounter, WrappedMemoryCounter};

    #[test]
    fn test_set_get_submatrix_basic() {
        // Create a 10x10 composite matrix
        let alloc_manager = WrappedAllocManager::new(AllocManager::new(WrappedMemoryCounter::new(
            MemoryCounter::new(10_000_000),
        )));
        let dir = Directory::Internal("test_composite_basic".to_string());
        let matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(5, 5, 10, 10, &dir, alloc_manager.clone());
        let wrapped = WrappedCompositeMatrix::new(matrix);

        // Fill with test values
        for i in 0..10 {
            for j in 0..10 {
                wrapped.set_mut_unchecked(i, j, NumberEntry((i * 10 + j) as f64));
            }
        }

        // Get a 3x3 submatrix starting at (2, 3)
        let submatrix = wrapped.get_submatrix(2, 3, 3, 3);

        // Verify values
        for i in 0..3 {
            for j in 0..3 {
                let expected = NumberEntry(((i + 2) * 10 + (j + 3)) as f64);
                let actual = submatrix.get_unchecked(i, j);
                assert_eq!(expected.0, actual.0, "Mismatch at ({}, {})", i, j);
            }
        }

        // Cleanup
        let _ = std::fs::remove_dir_all("test_composite_basic");
    }

    #[test]
    fn test_set_submatrix() {
        // Create a 10x10 composite matrix
        let alloc_manager = WrappedAllocManager::new(AllocManager::new(WrappedMemoryCounter::new(
            MemoryCounter::new(10_000_000),
        )));
        let dir = Directory::Internal("test_composite_set".to_string());
        let matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(5, 5, 10, 10, &dir, alloc_manager.clone());
        let wrapped = WrappedCompositeMatrix::new(matrix);

        // Initialize with zeros
        for i in 0..10 {
            for j in 0..10 {
                wrapped.set_mut_unchecked(i, j, NumberEntry(0.0));
            }
        }

        // Create a 3x3 submatrix with specific values
        let sub_dir = Directory::Internal("test_composite_sub".to_string());
        let sub_matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(3, 3, 3, 3, &sub_dir, alloc_manager.clone());
        let sub_wrapped = WrappedCompositeMatrix::new(sub_matrix);

        for i in 0..3 {
            for j in 0..3 {
                sub_wrapped.set_mut_unchecked(i, j, NumberEntry(100.0 + (i * 3 + j) as f64));
            }
        }

        // Set the submatrix at position (2, 3)
        wrapped.set_submatrix(2, 3, &sub_wrapped);

        // Verify that the submatrix was set correctly
        for i in 0..3 {
            for j in 0..3 {
                let expected = NumberEntry(100.0 + (i * 3 + j) as f64);
                let actual = wrapped.get_unchecked(i + 2, j + 3);
                assert_eq!(expected.0, actual.0, "Mismatch at ({}, {})", i + 2, j + 3);
            }
        }

        // Verify that other positions are still zero
        assert_eq!(wrapped.get_unchecked(0, 0).0, 0.0);
        assert_eq!(wrapped.get_unchecked(9, 9).0, 0.0);
        assert_eq!(wrapped.get_unchecked(1, 1).0, 0.0);

        // Cleanup
        let _ = std::fs::remove_dir_all("test_composite_set");
        let _ = std::fs::remove_dir_all("test_composite_sub");
    }

    #[test]
    fn test_set_get_submatrix_roundtrip() {
        // Create a 20x20 composite matrix
        let alloc_manager = WrappedAllocManager::new(AllocManager::new(WrappedMemoryCounter::new(
            MemoryCounter::new(10_000_000),
        )));
        let dir = Directory::Internal("test_composite_roundtrip".to_string());
        let matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(5, 5, 20, 20, &dir, alloc_manager.clone());
        let wrapped = WrappedCompositeMatrix::new(matrix);

        // Fill with test values
        for i in 0..20 {
            for j in 0..20 {
                wrapped.set_mut_unchecked(i, j, NumberEntry((i * 100 + j) as f64));
            }
        }

        // Extract a submatrix
        let submatrix = wrapped.get_submatrix(5, 7, 6, 8);

        // Create a new matrix and set the submatrix
        let dir2 = Directory::Internal("test_composite_roundtrip2".to_string());
        let matrix2: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(5, 5, 20, 20, &dir2, alloc_manager.clone());
        let wrapped2 = WrappedCompositeMatrix::new(matrix2);

        // Initialize with zeros
        for i in 0..20 {
            for j in 0..20 {
                wrapped2.set_mut_unchecked(i, j, NumberEntry(0.0));
            }
        }

        // Set the submatrix at the same position
        wrapped2.set_submatrix(5, 7, &submatrix);

        // Verify that the submatrix region matches
        for i in 5..11 {
            for j in 7..15 {
                let expected = wrapped.get_unchecked(i, j);
                let actual = wrapped2.get_unchecked(i, j);
                assert_eq!(expected.0, actual.0, "Mismatch at ({}, {})", i, j);
            }
        }

        // Cleanup
        let _ = std::fs::remove_dir_all("test_composite_roundtrip");
        let _ = std::fs::remove_dir_all("test_composite_roundtrip2");
    }

    #[test]
    #[should_panic(expected = "Submatrix dimensions exceed bounds of composite matrix")]
    fn test_get_submatrix_out_of_bounds() {
        let alloc_manager = WrappedAllocManager::new(AllocManager::new(WrappedMemoryCounter::new(
            MemoryCounter::new(10_000_000),
        )));
        let dir = Directory::Internal("test_composite_bounds".to_string());
        let matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(5, 5, 10, 10, &dir, alloc_manager);
        let wrapped = WrappedCompositeMatrix::new(matrix);

        // This should panic - trying to get an 8x8 submatrix starting at (5, 5)
        let _ = wrapped.get_submatrix(5, 5, 8, 8);

        let _ = std::fs::remove_dir_all("test_composite_bounds");
    }

    #[test]
    #[should_panic(expected = "Submatrix dimensions exceed bounds of composite matrix")]
    fn test_set_submatrix_out_of_bounds() {
        let alloc_manager = WrappedAllocManager::new(AllocManager::new(WrappedMemoryCounter::new(
            MemoryCounter::new(10_000_000),
        )));
        let dir = Directory::Internal("test_composite_set_bounds".to_string());
        let matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(5, 5, 10, 10, &dir, alloc_manager.clone());
        let wrapped = WrappedCompositeMatrix::new(matrix);

        // Create a 6x6 submatrix
        let sub_dir = Directory::Internal("test_composite_set_bounds_sub".to_string());
        let sub_matrix: CompositeMatrix<NumberEntry> =
            CompositeMatrix::new(3, 3, 6, 6, &sub_dir, alloc_manager);
        let sub_wrapped = WrappedCompositeMatrix::new(sub_matrix);

        // This should panic - trying to set a 6x6 submatrix at position (7, 7)
        wrapped.set_submatrix(7, 7, &sub_wrapped);

        let _ = std::fs::remove_dir_all("test_composite_set_bounds");
        let _ = std::fs::remove_dir_all("test_composite_set_bounds_sub");
    }
}
