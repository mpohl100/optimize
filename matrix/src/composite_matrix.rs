use crate::directory::Directory;
use crate::mat::WrappedMatrix;
use crate::persistable_matrix::{PersistableMatrix, PersistableValue};

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
    /// Panics if set_mut_unchecked fails
    #[must_use]
    pub fn new(
        slice_x: usize,
        slice_y: usize,
        rows: usize,
        cols: usize,
        directory: &Directory,
    ) -> Self {
        let matrices = WrappedMatrix::new(rows / slice_x, cols / slice_y);
        for i in 0..(rows / slice_x) {
            for j in 0..(cols / slice_y) {
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
        let binding = self.matrices.mat();
        let mut matrices = binding.lock().unwrap();
        let persistable_matrix = matrices.get_mut_unchecked(matrix_x, matrix_y);
        persistable_matrix.set_mut_unchecked(within_x, within_y, value);
    }
}
