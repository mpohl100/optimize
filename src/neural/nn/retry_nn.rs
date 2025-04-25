use super::shape::NeuralNetworkShape;
use super::nn_trait::WrappedNeuralNetwork;

use crate::gen::pheno::annotated_nn_shape::AnnotatedNeuralNetworkShape;
use crate::neural::nn::shape::LayerShape;
use crate::neural::nn::shape::LayerType;
use crate::neural::nn::neuralnet::ClassicNeuralNetwork;
use crate::neural::nn::nn_trait::NeuralNetwork;
use crate::neural::nn::directory::Directory;

#[derive(Debug)]
pub struct RetryNeuralNetwork{
    primary_nn: WrappedNeuralNetwork,
    backup_nn: WrappedNeuralNetwork,
    // The shape of the neural network that it should pretend to have to the outside world
    shape: NeuralNetworkShape,
    model_directory: Directory,
}

impl RetryNeuralNetwork {
    pub fn new(shape: NeuralNetworkShape, levels: i32, internal_model_directory: String) -> Self
    {
        let actual_shape = add_internal_dimensions(shape.clone());
        let primary_nn = WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(actual_shape.clone(), append_dir(internal_model_directory.clone(), "primary"))));
        let backup_nn = match levels {
            1..=i32::MAX => WrappedNeuralNetwork::new(Box::new(RetryNeuralNetwork::new(shape.clone(), levels - 1, append_dir(internal_model_directory.clone(), "backup")))),
            0 => WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(shape.clone(), append_dir(internal_model_directory.clone(), "backup")))),
            _ => panic!("Invalid level: {}", levels),
        };
        Self {
            primary_nn,
            backup_nn,
            shape,
            model_directory: Directory::Internal(internal_model_directory),
        }
    }

    pub fn from_disk(model_directory: String) -> WrappedNeuralNetwork {
        let primary_model_directory = append_dir(model_directory.clone(), "primary");
        let backup_model_directory = append_dir(model_directory.clone(), "backup");
        if std::path::Path::new(&primary_model_directory).exists() {
            let primary_nn = WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::from_disk(primary_model_directory).unwrap()));
            let backup_nn = RetryNeuralNetwork::from_disk(backup_model_directory);
            let shape = backup_nn.shape();
            return WrappedNeuralNetwork::new(Box::new(Self {
                primary_nn,
                backup_nn,
                shape,
                model_directory: Directory::User(model_directory),
            }));    
        }
        else {
            return WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::from_disk(model_directory).unwrap()));
        }
    }

    fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let primary_output = self.primary_nn.predict(input.clone());
        // if the last value in primary output is as close to zero as some tolerance, then we need to use the backup neural network
        if primary_output[primary_output.len() - 1].abs() < 0.05 {
            self.backup_nn.predict(input)
        } else {
            // return the primary output despite the last internal value
            primary_output[0..primary_output.len() - 1].to_vec()
        }
    }
}

fn add_internal_dimensions(shape: NeuralNetworkShape) -> NeuralNetworkShape {
    // Add internal dimensions to the shape
    let mut annotated_shape = AnnotatedNeuralNetworkShape::new(shape.clone());
    let first_layer = shape.layers.first().unwrap();

    // Add internal dimensions to the first layer
    let internal_layer = first_layer.clone();
    let new_dense_layer_type = LayerShape{ layer_type: LayerType::Dense {
        input_size: internal_layer.input_size(),
        output_size: internal_layer.output_size() + 1,
    }, activation: internal_layer.activation.clone() };
    annotated_shape.change_layer(0, new_dense_layer_type);

    // Add internal dimensions to the rest of the layers
    for (i, layer) in shape.layers.iter().skip(1).enumerate() {
        // Add internal dimensions to the layer
        let internal_layer = layer.clone();
        let new_dense_layer_type = LayerShape{ layer_type: LayerType::Dense {
            input_size: internal_layer.input_size() + 1,
            output_size: internal_layer.output_size() + 1,
        }, activation: internal_layer.activation.clone() };
        annotated_shape.change_layer(i, new_dense_layer_type);
    }
    annotated_shape.to_neural_network_shape()
}

fn append_dir(model_directory: String, subdir: &str) -> String {
    let mut path = model_directory.clone();
    path.push_str("/");
    path.push_str(subdir);
    path
}

impl NeuralNetwork for RetryNeuralNetwork {

    fn predict(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.forward(input)
    }

    fn shape(&self) -> NeuralNetworkShape {
        self.shape.clone()
    }

    fn save(&mut self, user_model_directory: String) -> Result<(), Box<dyn std::error::Error>> {
        let primary_user_model_directory = append_dir(user_model_directory.clone(), "primary");
        self.primary_nn.save(primary_user_model_directory)?;
        let backup_user_model_directory = append_dir(user_model_directory, "backup");
        self.backup_nn.save(backup_user_model_directory)?;
        Ok(())
    }

    fn get_model_directory(&self) -> Directory {
        self.model_directory.clone()
    }
}