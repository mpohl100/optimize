use alloc::allocatable::WrappedAllocatableTrait;
use matrix::composite_matrix::CompositeMatrix;
use matrix::persist::traits::PersistableValue;
use rayon::iter::ParallelIterator;

pub fn forward_impl<
    WeightT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Sync + 'static,
    BiasT: Default + Clone + PersistableValue + From<f64> + std::fmt::Debug + Sync + 'static,
>(
    mat: &CompositeMatrix<WeightT>,
    inputs: &[f64],
    biases: &CompositeMatrix<BiasT>,
    weight_value_retriever: fn(&WeightT) -> f64,
    bias_value_retriever: fn(&BiasT) -> f64,
) -> Vec<f64> {
    let mut outputs = vec![0.0; mat.rows()];
    mat.matrices().mat().lock().unwrap().iter_mut().enumerate().for_each(|(i, row)| {
        let num_rows = if (i + 1) * mat.get_slice_num_rows() > mat.rows() {
            mat.rows() - i * mat.get_slice_num_rows()
        } else {
            mat.get_slice_num_rows()
        };
        let mut local_outputs = vec![0.0; num_rows];

        let num_cols = if (i + 1) * mat.get_slice_num_cols() > mat.cols() {
            mat.cols() - i * mat.get_slice_num_cols()
        } else {
            mat.get_slice_num_cols()
        };
        let local_inputs =
            &inputs[i * mat.get_slice_num_cols()..i * mat.get_slice_num_cols() + num_cols];
        for matrix in row.iter_mut() {
            let mut alloc_manager = mat.get_alloc_manager();
            alloc_manager.allocate(matrix);
            matrix.mark_for_use();
            let col_outputs = matrix
                .mat()
                .lock()
                .unwrap()
                .mat()
                .unwrap()
                .mat()
                .lock()
                .unwrap()
                .par_iter()
                .map(|row| {
                    let value = row
                        .iter()
                        .zip(local_inputs.iter())
                        .map(|(w, &x)| weight_value_retriever(w) * x)
                        .sum::<f64>();
                    value
                })
                .collect::<Vec<f64>>();
            matrix.free_from_use();
            for (k, val) in col_outputs.iter().enumerate() {
                local_outputs[k] += val;
            }
        }
        // add local biases
        let mut local_biases = biases.matrices().get_unchecked(i, 0);

        let mut alloc_manager = biases.get_alloc_manager();
        alloc_manager.allocate(&local_biases);
        local_biases.mark_for_use();
        local_biases
            .mat()
            .lock()
            .unwrap()
            .mat()
            .unwrap()
            .mat()
            .lock()
            .unwrap()
            .iter_mut()
            .enumerate()
            .for_each(|(bias_index, bias_row)| {
                for bias in bias_row.iter_mut() {
                    local_outputs[bias_index] += bias_value_retriever(bias);
                }
            });
        local_biases.free_from_use();

        for (k, val) in local_outputs.iter().enumerate() {
            outputs[i * mat.get_slice_num_rows() + k] += val;
        }
    });
    outputs
}
