//! Integration tests for the solver crate.

use neural::layer::dense_layer::MatrixParams;
use neural::nn::shape::{
    ActivationData, ActivationType, LayerShape, LayerType, NeuralNetworkShape,
};
use neural::training::training_params::TrainingParams;
use neural::utilities::util::{Utils, WrappedUtils};
use solver::neural_solver::NeuralSolver;

#[test]
#[ignore = "requires sufficiently large matrix dimensions for slice_rows/slice_cols; run manually"]
fn test_neural_solver_creation() {
    // Create a simple neural network shape
    let layers = vec![
        LayerShape {
            layer_type: LayerType::Dense {
                input_size: 10,
                output_size: 5,
                matrix_params: MatrixParams { slice_rows: 10, slice_cols: 10 },
            },
            activation: ActivationData::new(ActivationType::ReLU),
        },
        LayerShape {
            layer_type: LayerType::Dense {
                input_size: 5,
                output_size: 3,
                matrix_params: MatrixParams { slice_rows: 10, slice_cols: 10 },
            },
            activation: ActivationData::new(ActivationType::Sigmoid),
        },
    ];
    let shape = NeuralNetworkShape::new(layers);

    let levels = None;
    let pre_shape = None;
    let validation_split = 1.0;
    let learning_rate = 0.01;
    let epochs = 10;
    let tolerance = 0.001;
    let batch_size = 32;
    let use_adam = true;
    let sample_match_percentage = 0.9;

    let training_params = TrainingParams::new(
        shape.clone(),
        levels,
        pre_shape,
        validation_split,
        learning_rate,
        epochs,
        tolerance,
        batch_size,
        use_adam,
        sample_match_percentage,
    );
    let all_inputs = vec![vec![0.0; 10]; 100];
    let all_targets = vec![vec![0.0; 3]; 100];
    let utils = WrappedUtils::new(Utils::new(1_000_000_000, 4));
    // Create a neural solver with the shape
    let mut solver = NeuralSolver::new(shape, training_params, all_inputs, all_targets, utils);

    // Call the solve method (with minimal iterations for testing)
    solver.solve(1, false, 0.9);
}
