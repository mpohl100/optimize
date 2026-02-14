//! Integration tests for the solver crate.

use neural::layer::dense_layer::MatrixParams;
use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType, NeuralNetworkShape};
use solver::neural_solver::NeuralSolver;

#[test]
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

    // Create a neural solver with the shape
    let solver = NeuralSolver::new(shape);

    // Call the solve method (with minimal iterations for testing)
    solver.solve(1, false);
}
