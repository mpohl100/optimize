pub use crate::nn::layer::Layer;
pub use crate::nn::activation::Activation;
pub use crate::nn::neuralnet::shape::NeuralNetworkShape;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Box<dyn Activation>>,
    shape: NeuralNetworkShape,
}

impl NeuralNetwork {
    /// Creates a new `NeuralNetwork` from the given shape.
    pub fn new(shape: NeuralNetworkShape) -> Self {
        let mut network = NeuralNetwork {
            layers: Vec::new(),
            activations: Vec::new(),
            shape,
        };

        // Initialize layers and activations based on the provided shape.
        for layer_shape in &network.shape.layers {
            // Here you would instantiate the appropriate Layer and Activation objects.
            let layer = Box::new(DenseLayer::new(layer_shape.input_size(), layer_shape.output_size()));
            let activation = match layer_shape.activation {
                ActivationType::ReLU => Box::new(ReLUActivation) as Box<dyn Activation>,
                ActivationType::Sigmoid => Box::new(SigmoidActivation) as Box<dyn Activation>,
            };

            network.add_activation_and_layer(activation, layer);
        }

        network
    }

    /// Adds an activation and a layer to the neural network.
    fn add_activation_and_layer(&mut self, activation: Box<dyn Activation>, layer: Box<dyn Layer>) {
        self.activations.push(activation);
        self.layers.push(layer);
    }

    /// Performs a forward pass through the network with the given input.
    pub fn forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut output = input;
        for (layer, activation) in self.layers.iter().zip(&self.activations) {
            output = layer.forward(&output);
            output = activation.forward(&output);
        }
        output
    }

    /// Performs a backward pass through the network with the given output gradient.
    pub fn backward(&mut self, grad_output: Vec<f64>) {
        let mut grad = grad_output;
        for (layer, activation) in self.layers.iter_mut().rev().zip(self.activations.iter_mut().rev()) {
            grad = activation.backward(&grad);
            grad = layer.backward(&grad);
        }
    }

    /// Trains the neural network using the given inputs, targets, learning rate, and number of epochs.
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets) {
                // Forward pass
                let output = self.forward(input.clone());

                // Calculate loss gradient (e.g., mean squared error)
                let mut grad_output: Vec<f64> = output.iter()
                    .zip(target)
                    .map(|(o, t)| o - t)
                    .collect();

                // Backward pass
                self.backward(grad_output);

                // Update weights
                for layer in &mut self.layers {
                    layer.update_weights(learning_rate);
                }
            }
        }
    }

    /// Makes a prediction based on a single input by performing a forward pass.
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input)
    }

    /// Returns the input size of the first layer in the network.
    pub fn input_size(&self) -> usize {
        self.shape.layers.first().map_or(0, |layer| layer.input_size())
    }

    /// Returns the output size of the last layer in the network.
    pub fn output_size(&self) -> usize {
        self.shape.layers.last().map_or(0, |layer| layer.output_size())
    }
}
