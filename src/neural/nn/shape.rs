use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

/// Enum representing the type of layer in a neural network.
/// Each variant includes the input size and output size of the layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// A fully connected (dense) layer with specified input and output sizes.
    Dense { input_size: usize, output_size: usize },
}

/// Enum representing the type of activation function used in a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// `ReLU` (Rectified Linear Unit) activation function.
    ReLU,
    /// Sigmoid activation function.
    Sigmoid,
    /// Tanh (Hyperbolic Tangent) activation function.
    Tanh,
    /// Softmax activation function.
    Softmax,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationData {
    activation_type: ActivationType,
    temperature: Option<f64>,
}

impl ActivationData {
    #[must_use] pub const fn new(activation_type: ActivationType) -> Self {
        Self { activation_type, temperature: None }
    }

    #[must_use] pub const fn new_softmax(temperature: f64) -> Self {
        Self { activation_type: ActivationType::Softmax, temperature: Some(temperature) }
    }

    #[must_use] pub fn is_valid(&self) -> bool {
        match self.activation_type {
            ActivationType::Softmax => {
                self.temperature.is_some() && self.temperature.unwrap() > 0.0
            },
            _ => self.temperature.is_none(),
        }
    }

    #[must_use] pub const fn activation_type(&self) -> ActivationType {
        self.activation_type
    }

    #[must_use] pub const fn temperature(&self) -> Option<f64> {
        self.temperature
    }
}

/// Struct representing the shape and configuration of a neural network layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerShape {
    /// The type of the layer (e.g., Dense) with input and output sizes.
    pub layer_type: LayerType,
    /// The activation function used in the layer (e.g., `ReLU`, Sigmoid).
    pub activation: ActivationData,
}

impl LayerShape {
    /// Returns the input size of the layer.
    #[must_use] pub const fn input_size(&self) -> usize {
        match self.layer_type {
            LayerType::Dense { input_size, .. } => input_size,
        }
    }

    /// Returns the output size of the layer.
    #[must_use] pub const fn output_size(&self) -> usize {
        match self.layer_type {
            LayerType::Dense { output_size, .. } => output_size,
        }
    }

    /// Returns the type of the layer.
    #[must_use] pub fn layer_type(&self) -> LayerType {
        self.layer_type.clone()
    }

    /// Checks if the layer shape is valid.
    ///
    /// # Returns
    ///
    /// * `true` if both input size and output size are greater than zero.
    /// * `false` otherwise.
    #[must_use] pub fn is_valid(&self) -> bool {
        self.input_size() > 0 && self.output_size() > 0 && self.activation.is_valid()
    }
}

/// Struct representing the shape and configuration of an entire neural network.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetworkShape {
    /// A vector of `LayerShape` structs, representing each layer in the neural network.
    pub layers: Vec<LayerShape>,
}

impl NeuralNetworkShape {
    /// Creates a new `NeuralNetworkShape` with the given layers.
    #[must_use] pub const fn new(layers: Vec<LayerShape>) -> Self {
        Self { layers }
    }

    /// Creates a new `NeuralNetworkShape` with the given layers from disk.
    #[must_use] pub fn from_disk(model_directory: String) -> Option<Self> {
        let path = format!("{model_directory}/shape.yaml");
        if !std::path::Path::new(&path).exists() {
            return None;
        }
        let file = File::open(path).unwrap();
        let shape: Self = serde_yaml::from_reader(file).unwrap();
        Some(shape)
    }

    // Creates a new `NeuralNetworkShape` with the given layers from file.
    #[must_use] pub fn from_file(file_name: String) -> Self {
        let file = File::open(file_name).unwrap();
        let shape: Self = serde_yaml::from_reader(file).unwrap();
        shape
    }

    /// Checks if the neural network shape is valid.
    ///
    /// # Returns
    ///
    /// * `true` if all layers are valid and the input size of each layer matches
    ///   the output size of the previous layer (except for the first layer).
    /// * `false` otherwise.
    #[must_use] pub fn is_valid(&self) -> bool {
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

    pub fn to_yaml(
        &self,
        model_directory: String,
    ) {
        // Create file in shape.yaml in model_directory and put the yaml there
        // ensure model_directory exists
        if std::fs::metadata(&model_directory).is_err() {
            std::fs::create_dir_all(&model_directory).unwrap();
        }
        let path = format!("{model_directory}/shape.yaml");
        let mut file = File::create(path).unwrap();
        let yaml = serde_yaml::to_string(self).unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
    }

    /// Returns the layer at the specified index.
    #[must_use] pub fn get_layer(
        &self,
        index: usize,
    ) -> LayerShape {
        self.layers[index].clone()
    }

    /// Returns the number of layers in the neural network shape.
    #[must_use] pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Adds a new layer at the specified position.
    pub fn add_layer(
        &mut self,
        position: usize,
        layer: LayerShape,
    ) {
        self.layers.insert(position, layer);
    }

    /// Changes layer at the specified position.
    pub fn change_layer(
        &mut self,
        position: usize,
        layer: LayerShape,
    ) {
        self.layers[position] = layer;
    }

    /// Cut out a subnetwork from the neural network shape.
    #[must_use] pub fn cut_out(
        &self,
        start: usize,
        end: usize,
    ) -> Self {
        // check that start are within bounds
        assert!(start < self.layers.len(), "Start index out of bounds");

        // check that end are within bounds
        assert!(end <= self.layers.len(), "End index out of bounds");

        let layers = self.layers[start..end].to_vec();
        Self { layers }
    }

    #[must_use] pub fn merge(
        &self,
        other: Self,
        middle_activation_data: ActivationData,
    ) -> Self {
        let mut layers = self.layers.clone();

        let merge_layer_input_size = self.layers.last().unwrap().output_size();
        let merge_layer_output_size = other.layers.first().unwrap().input_size();

        let merge_layer = LayerShape {
            layer_type: LayerType::Dense {
                input_size: merge_layer_input_size,
                output_size: merge_layer_output_size,
            },
            activation: middle_activation_data,
        };
        layers.push(merge_layer);

        layers.extend(other.layers);
        let new_shape = Self { layers };
        assert!(new_shape.is_valid());
        new_shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_shape_validity() {
        let valid_layer = LayerShape {
            layer_type: LayerType::Dense { input_size: 10, output_size: 5 },
            activation: ActivationData::new(ActivationType::ReLU),
        };
        assert!(valid_layer.is_valid());

        let invalid_layer = LayerShape {
            layer_type: LayerType::Dense { input_size: 0, output_size: 5 },
            activation: ActivationData::new(ActivationType::Sigmoid),
        };
        assert!(!invalid_layer.is_valid());
    }

    #[test]
    fn test_neural_network_shape_validity() {
        let layers = vec![
            LayerShape {
                layer_type: LayerType::Dense { input_size: 10, output_size: 5 },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense { input_size: 5, output_size: 3 },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
        ];
        let network = NeuralNetworkShape { layers };
        assert!(network.is_valid());

        let invalid_layers = vec![
            LayerShape {
                layer_type: LayerType::Dense { input_size: 10, output_size: 5 },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense {
                    input_size: 4, // Mismatch here
                    output_size: 3,
                },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
        ];
        let invalid_network = NeuralNetworkShape { layers: invalid_layers };
        assert!(!invalid_network.is_valid());
    }
}
