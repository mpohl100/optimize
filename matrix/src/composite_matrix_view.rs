use crate::composite_matrix_buffer::WrappedCompositeMatrixBuffer;
use crate::matrix_view::mat_apply;
use crate::matrix_view::MatrixView;
use crate::matrix_view::{AccessorType, ApplierFunc, WrappedAccessorType};

use std::sync::{Arc, Mutex};

pub struct CompositeMatrixView<T: Default + Clone, BufferT: Default + Clone> {
    buffer: WrappedCompositeMatrixBuffer<BufferT>,
    accessor_type: WrappedAccessorType<T, BufferT>,
}

impl<T: Default + Clone, BufferT: Default + Clone> CompositeMatrixView<T, BufferT> {
    /// Applies an accessor function to the given buffer value.
    /// # Arguments
    /// * `buffer_val` - The buffer value to be accessed.
    /// # Returns
    /// The result of applying the accessor function to the buffer value.
    /// # Panics
    /// Panics if the accessor type is a setter instead of a getter.
    pub fn apply_accessor(
        &self,
        buffer_val: &BufferT,
    ) -> T {
        let accessor_type = self.accessor_type.accessor_type.lock().unwrap();
        match &*accessor_type {
            AccessorType::Get(getter) => getter.lock().unwrap()(buffer_val),
            AccessorType::Set(_) => panic!("Cannot apply a setter as an accessor"),
        }
    }

    /// Applies a mutator function to the given buffer value.
    /// # Arguments
    /// * `buffer_val` - The buffer value to be mutated.
    /// * `value` - The value to be applied to the buffer using the mutator
    /// # Panics
    /// Panics if the accessor type is a getter instead of a setter.
    pub fn apply_mutator(
        &self,
        buffer_val: &mut BufferT,
        value: T,
    ) {
        let accessor_type = self.accessor_type.accessor_type.lock().unwrap();
        match &*accessor_type {
            AccessorType::Set(setter) => setter.lock().unwrap()(buffer_val, value),
            AccessorType::Get(_) => panic!("Cannot apply a getter as a mutator"),
        }
    }
}

impl<T: Default + Clone, BufferT: Default + Clone> CompositeMatrixView<T, BufferT> {
    /// Returns the shape (nrows, ncols) of the matrix view.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.buffer.shape()
    }

    #[must_use]
    pub fn num_sub_matrices(&self) -> (usize, usize) {
        self.buffer.num_sub_matrices()
    }

    /// Returns the value at (row, col).
    #[must_use]
    pub fn get(
        &self,
        row: usize,
        col: usize,
    ) -> T {
        self.apply_accessor(&self.buffer.get_val(row, col).unwrap_or_default())
    }

    /// Returns a view of the sub-matrix at the specified position.
    /// # Arguments
    /// * `num_row` - The row index of the sub-matrix.
    /// * `num_col` - The column index of the sub-matrix.
    /// # Returns
    /// A `MatrixView` representing the sub-matrix at the specified position.
    /// # Panics
    /// Panics if the specified sub-matrix indices are out of bounds.
    #[must_use]
    pub fn get_sub_matrix_view(
        &self,
        num_row: usize,
        num_col: usize,
    ) -> MatrixView<T, BufferT> {
        let sub_buffer = self.buffer.get_sub_matrix_buffer(num_row, num_col);
        let accessor_type = self.accessor_type.accessor_type.lock().unwrap().clone();
        MatrixView::new(sub_buffer.unwrap(), accessor_type)
    }
}

impl<T: Default + Clone, BufferT: Default + Clone> CompositeMatrixView<T, BufferT> {
    /// Creates a new `CompositeMatrixView` with the specified number of rows and columns.
    #[must_use]
    pub fn new(
        wrapped_matrix_buffer: WrappedCompositeMatrixBuffer<BufferT>,
        accessor_type: AccessorType<T, BufferT>,
    ) -> Self {
        Self {
            buffer: wrapped_matrix_buffer,
            accessor_type: WrappedAccessorType {
                accessor_type: Arc::new(Mutex::new(accessor_type)),
            },
        }
    }
}

/// Applies a function element-wise to two input matrices and stores the result in a third matrix.
/// # Arguments
/// * `result` - The matrix view where the results will be stored.
/// * `first` - The first input matrix view.
/// * `second` - The second input matrix view.
/// * `applier_func` - The function to apply to corresponding elements from the input matrices
/// # Panics
/// Panics if the shapes of the input matrices are not compatible for the operation.
pub fn composite_mat_apply<
    ResultT: Default + Clone + std::ops::AddAssign,
    ResultBufferT: Default + Clone,
    FirstT: Default + Clone,
    FirstBufferT: Default + Clone,
    SecondT: Default + Clone,
    SecondBufferT: Default + Clone,
>(
    result: &mut CompositeMatrixView<ResultT, ResultBufferT>,
    first: &CompositeMatrixView<FirstT, FirstBufferT>,
    second: &CompositeMatrixView<SecondT, SecondBufferT>,
    applier_func: &ApplierFunc<ResultT, FirstT, SecondT>,
) {
    for num_row in 0..result.num_sub_matrices().0 {
        for num_col in 0..result.num_sub_matrices().1 {
            for k in 0..first.num_sub_matrices().1 {
                let mut result_sub_matrix = result.get_sub_matrix_view(num_row, num_col);
                let first_sub_matrix = first.get_sub_matrix_view(num_row, k);
                let second_sub_matrix = second.get_sub_matrix_view(k, num_col);
                mat_apply(
                    &mut result_sub_matrix,
                    &first_sub_matrix,
                    &second_sub_matrix,
                    applier_func,
                );
            }
        }
    }
}

pub fn composite_mat_mult<
    ResultBufferT: Default + Clone,
    FirstBufferT: Default + Clone,
    SecondBufferT: Default + Clone,
>(
    result: &mut CompositeMatrixView<f64, ResultBufferT>,
    first: &CompositeMatrixView<f64, FirstBufferT>,
    second: &CompositeMatrixView<f64, SecondBufferT>,
) {
    let func = |f: &f64, s: &f64| f * s;
    let applier_func = Arc::new(Mutex::new(Box::new(func) as Box<dyn FnMut(&f64, &f64) -> f64>));
    composite_mat_apply(result, first, second, &applier_func);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_matrix_view() {
        let buffer = WrappedCompositeMatrixBuffer::<f64>::new(2, 2, 2, 2);
        let accessor_type = AccessorType::Get(Arc::new(Mutex::new(Box::new(|x: &f64| *x))));
        let matrix_view = CompositeMatrixView::new(buffer, accessor_type);

        // Test getting values (should be default 0.0)
        assert_eq!(matrix_view.get(0, 0), 0.0);
        assert_eq!(matrix_view.get(1, 1), 0.0);
    }

    #[derive(Default, Clone)]
    struct MyVals {
        pub result: f64,
        pub first: f64,
        pub second: f64,
    }

    #[test]
    fn test_matrix_multiplication() {
        let result_buffer = WrappedCompositeMatrixBuffer::<MyVals>::new(2, 2, 2, 2);
        let result_accessor =
            AccessorType::Set(Arc::new(Mutex::new(Box::new(|buffer: &mut MyVals, value: f64| {
                // remember to add to the existing value
                buffer.result += value;
            }))));
        let mut result_view = CompositeMatrixView::new(result_buffer.clone(), result_accessor);
        let first_accessor =
            AccessorType::Get(Arc::new(Mutex::new(Box::new(|x: &MyVals| x.first))));
        let first_view = CompositeMatrixView::new(result_buffer.clone(), first_accessor);
        let second_accessor =
            AccessorType::Get(Arc::new(Mutex::new(Box::new(|x: &MyVals| x.second))));
        let second_view = CompositeMatrixView::new(result_buffer, second_accessor);

        composite_mat_mult(&mut result_view, &first_view, &second_view);
    }
}
