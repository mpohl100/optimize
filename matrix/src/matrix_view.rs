use crate::matrix_buffer::WrappedMatrixBuffer;

use std::sync::{Arc, Mutex};

pub type AccessorFunc<T, BufferT> = Arc<Mutex<dyn FnMut(&BufferT) -> T + Send + Sync>>;
pub type MutatorFunc<T, BufferT> = Arc<Mutex<dyn FnMut(&mut BufferT, T)>>;

pub enum AccessorType<T: Default + Clone, BufferT: Default + Clone> {
    Get(AccessorFunc<T, BufferT>),
    Set(MutatorFunc<T, BufferT>),
}

pub enum IterationType {
    RowWise,
    ColumnWise,
    Block,
}

pub struct MatrixView<T: Default + Clone, BufferT: Default + Clone> {
    buffer: WrappedMatrixBuffer<BufferT>,
    accessor_type: AccessorType<T, BufferT>,
    iteration_type: IterationType,
}

impl<T: Default + Clone, BufferT: Default + Clone> MatrixView<T, BufferT> {
    /// Creates a new `MatrixView` with the specified number of rows and columns.
    #[must_use]
    pub const fn new(
        wrapped_matrix_buffer: WrappedMatrixBuffer<BufferT>,
        accessor_type: AccessorType<T, BufferT>,
        iteration_type: IterationType,
    ) -> Self {
        Self { buffer: wrapped_matrix_buffer, accessor_type, iteration_type }
    }
}

pub type ApplierFunc<ResultT, FirstT, SecondT> =
    Arc<Mutex<dyn FnMut(&FirstT, &SecondT) -> ResultT + Send + Sync>>;

pub fn mat_apply<
    ResultT: Default + Clone,
    ResultBufferT: Default + Clone,
    FirstT: Default + Clone,
    FirstBufferT: Default + Clone,
    SecondT: Default + Clone,
    SecondBufferT: Default + Clone,
>(
    result: &mut MatrixView<ResultT, ResultBufferT>,
    first: MatrixView<FirstT, FirstBufferT>,
    second: MatrixView<SecondT, SecondBufferT>,
    applier_func: ApplierFunc<ResultT, FirstT, SecondT>,
) {
    // Placeholder for matrix multiplication logic
    // This function would perform matrix multiplication using the provided views
    for (r, f_row, s_col) in result.iter_mut_block().zip(first.iter_row()).zip(second.iter_col()) {
        for (f_cell, s_cell) in f_row.zip(s_col) {
            r += applier_func.lock().unwrap()(&f_cell, &s_cell);
        }
    }
}
