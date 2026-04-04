use crate::matrix_buffer::WrappedMatrixBuffer;

use std::sync::{Arc, Mutex};

pub type AccessorFunc<T, BufferT> = Arc<Mutex<dyn FnMut(&BufferT) -> T + Send + Sync>>;
pub type MutatorFunc<T, BufferT> = Arc<Mutex<dyn FnMut(&mut BufferT, T)>>;

#[derive(Clone)]
pub enum AccessorType<T: Default + Clone, BufferT: Default + Clone> {
    Get(AccessorFunc<T, BufferT>),
    Set(MutatorFunc<T, BufferT>),
}

pub struct WrappedAccessorType<T: Default + Clone, BufferT: Default + Clone> {
    pub accessor_type: Arc<Mutex<AccessorType<T, BufferT>>>,
}

pub struct MatrixView<T: Default + Clone, BufferT: Default + Clone> {
    buffer: WrappedMatrixBuffer<BufferT>,
    accessor_type: WrappedAccessorType<T, BufferT>,
}

impl<T: Default + Clone, BufferT: Default + Clone> MatrixView<T, BufferT> {
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

impl<T: Default + Clone, BufferT: Default + Clone> MatrixView<T, BufferT> {
    /// Returns the shape (nrows, ncols) of the matrix view.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        // Placeholder: actual implementation depends on buffer API
        self.buffer.shape()
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
}

impl<T: Default + Clone, BufferT: Default + Clone> MatrixView<T, BufferT> {
    /// Creates a new `MatrixView` with the specified number of rows and columns.
    #[must_use]
    pub fn new(
        wrapped_matrix_buffer: WrappedMatrixBuffer<BufferT>,
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

pub type ApplierFunc<ResultT, FirstT, SecondT> =
    Arc<Mutex<Box<dyn FnMut(&FirstT, &SecondT) -> ResultT>>>;

/// Applies a function element-wise to two input matrices and stores the result in a third matrix.
/// # Arguments
/// * `result` - The matrix view where the results will be stored.
/// * `first` - The first input matrix view.
/// * `second` - The second input matrix view.
/// * `applier_func` - The function to apply to corresponding elements from the input matrices
/// # Panics
/// Panics if the shapes of the input matrices are not compatible for the operation.
pub fn mat_apply<
    ResultT: Default + Clone + std::ops::AddAssign,
    ResultBufferT: Default + Clone,
    FirstT: Default + Clone,
    FirstBufferT: Default + Clone,
    SecondT: Default + Clone,
    SecondBufferT: Default + Clone,
>(
    result: &mut MatrixView<ResultT, ResultBufferT>,
    first: &MatrixView<FirstT, FirstBufferT>,
    second: &MatrixView<SecondT, SecondBufferT>,
    applier_func: &ApplierFunc<ResultT, FirstT, SecondT>,
) {
    for num_row in 0..result.shape().0 {
        for num_col in 0..result.shape().1 {
            let mut cell_value = ResultT::default();
            for k in 0..first.shape().1 {
                let f_cell = first.get(num_row, k);
                let s_cell = second.get(k, num_col);
                cell_value += applier_func.lock().unwrap()(&f_cell, &s_cell);
            }
            result.apply_mutator(
                &mut result.buffer.get_val(num_row, num_col).unwrap_or_default(),
                cell_value,
            );
        }
    }
}

pub fn mat_mult<
    ResultBufferT: Default + Clone,
    FirstBufferT: Default + Clone,
    SecondBufferT: Default + Clone,
>(
    result: &mut MatrixView<f64, ResultBufferT>,
    first: &MatrixView<f64, FirstBufferT>,
    second: &MatrixView<f64, SecondBufferT>,
) {
    let func = |f: &f64, s: &f64| f * s;
    let applier_func = Arc::new(Mutex::new(Box::new(func) as Box<dyn FnMut(&f64, &f64) -> f64>));
    mat_apply(result, first, second, &applier_func);
}

#[cfg(test)]
mod tests {
    use super::mat_mult;
    use crate::matrix_buffer::WrappedMatrixBuffer;
    use crate::matrix_view::{AccessorType, MatrixView};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_matrix_view() {
        let buffer = WrappedMatrixBuffer::<f64>::new(2, 2);
        let accessor_type = AccessorType::Get(Arc::new(Mutex::new(Box::new(|x: &f64| *x))));
        let matrix_view = MatrixView::new(buffer, accessor_type);

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
        let result_buffer = WrappedMatrixBuffer::<MyVals>::new(2, 2);
        let result_accessor =
            AccessorType::Set(Arc::new(Mutex::new(Box::new(|buffer: &mut MyVals, value: f64| {
                buffer.result = value;
            }))));
        let mut result_view = MatrixView::new(result_buffer.clone(), result_accessor);
        let first_accessor =
            AccessorType::Get(Arc::new(Mutex::new(Box::new(|x: &MyVals| x.first))));
        let first_view = MatrixView::new(result_buffer.clone(), first_accessor);
        let second_accessor =
            AccessorType::Get(Arc::new(Mutex::new(Box::new(|x: &MyVals| x.second))));
        let second_view = MatrixView::new(result_buffer, second_accessor);

        mat_mult(&mut result_view, &first_view, &second_view);
    }
}
