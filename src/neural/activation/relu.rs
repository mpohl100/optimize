use crate::neural::nn::shape::ActivationData;
use crate::neural::nn::shape::ActivationType;

use super::activate::ActivationTrait;

/// ReLU (Rectified Linear Unit) activation function.
#[derive(Debug, Clone)]
pub struct ReLU;

impl ActivationTrait for ReLU {
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect()
    }

    fn backward(&self, grad_output: &[f64]) -> Vec<f64> {
        grad_output
            .iter()
            .map(|&output| if output > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }

    fn get_activation_data(&self) -> ActivationData {
        ActivationData::new(ActivationType::ReLU)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = ReLU;
        let input = vec![-1.0, 0.0, 1.0];
        let output = relu.forward(&input);
        // print output
        println!("{:?}", output);
        assert_eq!(output, vec![0.0, 0.0, 1.0]);

        let grad_output = vec![-0.5, 0.0, 0.5];
        let grad_input = relu.backward(&grad_output);
        // print grad_input
        println!("{:?}", grad_input);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0]);
    }
}
