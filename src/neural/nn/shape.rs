use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

/// Enum representing the type of layer in a neural network.
/// Each variant includes the input size and output size of the layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerType {
    /// A fully connected (dense) layer with specified input and output sizes.
    Dense {
        input_size: usize,
        output_size: usize,
    },
}

/// Enum representing the type of activation function used in a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// ReLU (Rectified Linear Unit) activation function.
    ReLU,
    /// Sigmoid activation function.
    Sigmoid,
    /// Tanh (Hyperbolic Tangent) activation function.
    Tanh,
}

/// Struct representing the shape and configuration of a neural network layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerShape {
    /// The type of the layer (e.g., Dense) with input and output sizes.
    pub layer_type: LayerType,
    /// The activation function used in the layer (e.g., ReLU, Sigmoid).
    pub activation: ActivationType,
}

impl LayerShape {
    /// Returns the input size of the layer.
    pub fn input_size(&self) -> usize {
        match self.layer_type {
            LayerType::Dense { input_size, .. } => input_size,
        }
    }

    /// Returns the output size of the layer.
    pub fn output_size(&self) -> usize {
        match self.layer_type {
            LayerType::Dense { output_size, .. } => output_size,
        }
    }

    /// Checks if the layer shape is valid.
    ///
    /// # Returns
    ///
    /// * `true` if both input size and output size are greater than zero.
    /// * `false` otherwise.
    pub fn is_valid(&self) -> bool {
        self.input_size() > 0 && self.output_size() > 0
    }
}

/// Struct representing the shape and configuration of an entire neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkShape {
    /// A vector of `LayerShape` structs, representing each layer in the neural network.
    pub layers: Vec<LayerShape>,
}

impl NeuralNetworkShape {
    /// Checks if the neural network shape is valid.
    ///
    /// # Returns
    ///
    /// * `true` if all layers are valid and the input size of each layer matches
    ///   the output size of the previous layer (except for the first layer).
    /// * `false` otherwise.
    pub fn is_valid(&self) -> bool {
        if self.layers.is_empty() {
            return false;
        }

        // Check if each layer is valid
        for layer in &self.layers {
            if !layer.is_valid() {
                return false;
            }
        }

        // Check consistency between consecutive layers
        for window in self.layers.windows(2) {
            let (prev, next) = (&window[0], &window[1]);
            if prev.output_size() != next.input_size() {
                return false;
            }
        }

        true
    }

    pub fn to_yaml(&self, model_directory: String) {
        let yaml = serde_yaml::to_string(self).unwrap();
        // Create file in shape.yaml in model_directory and put the yaml there
        let path = format!("{}/shape.yaml", model_directory);
        let mut file = File::create(path).unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_shape_validity() {
        let valid_layer = LayerShape {
            layer_type: LayerType::Dense {
                input_size: 10,
                output_size: 5,
            },
            activation: ActivationType::ReLU,
        };
        assert!(valid_layer.is_valid());

        let invalid_layer = LayerShape {
            layer_type: LayerType::Dense {
                input_size: 0,
                output_size: 5,
            },
            activation: ActivationType::Sigmoid,
        };
        assert!(!invalid_layer.is_valid());
    }

    #[test]
    fn test_neural_network_shape_validity() {
        let layers = vec![
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 10,
                    output_size: 5,
                },
                activation: ActivationType::ReLU,
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 5,
                    output_size: 3,
                },
                activation: ActivationType::Sigmoid,
            },
        ];
        let network = NeuralNetworkShape { layers };
        assert!(network.is_valid());

        let invalid_layers = vec![
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 10,
                    output_size: 5,
                },
                activation: ActivationType::ReLU,
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 4, // Mismatch here
                    output_size: 3,
                },
                activation: ActivationType::Sigmoid,
            },
        ];
        let invalid_network = NeuralNetworkShape {
            layers: invalid_layers,
        };
        assert!(!invalid_network.is_valid());
    }
}
