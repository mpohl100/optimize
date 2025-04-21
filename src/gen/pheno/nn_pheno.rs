use super::{
    nn_mutater::fetch_activation_data, nn_mutater::NeuralNetworkMutater, rng_wrapper::RealRng,
};
use crate::evol::phenotype::Phenotype;
use crate::evol::rng::RandomNumberGenerator;
use crate::evol::strategy::Adjust;
use crate::neural::nn::neuralnet::TrainableClassicNeuralNetwork;
use crate::neural::nn::nn_trait::NeuralNetwork;
use crate::neural::nn::shape::NeuralNetworkShape;

use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct NeuralNetworkPhenotype {
    nn: Arc<Mutex<TrainableClassicNeuralNetwork>>,
    left_half_shape: Option<NeuralNetworkShape>,
    right_half_shape: Option<NeuralNetworkShape>,
    nb_mutates: usize,
}

impl Clone for NeuralNetworkPhenotype {
    fn clone(&self) -> Self {
        Self {
            nn: Arc::new(Mutex::new(self.get_nn())),
            left_half_shape: self.left_half_shape.clone(),
            right_half_shape: self.right_half_shape.clone(),
            nb_mutates: self.nb_mutates,
        }
    }
}

impl NeuralNetworkPhenotype {
    pub fn new(nn: TrainableClassicNeuralNetwork) -> Self {
        Self {
            nn: Arc::new(Mutex::new(nn)),
            left_half_shape: None,
            right_half_shape: None,
            nb_mutates: 0,
        }
    }

    pub fn get_nn(&self) -> TrainableClassicNeuralNetwork {
        self.nn.lock().unwrap().clone()
    }

    pub fn set_nn(&mut self, nn: TrainableClassicNeuralNetwork) {
        *self.nn.lock().unwrap() = nn;
    }

    fn set_left_half_shape(&mut self, shape: NeuralNetworkShape) {
        self.left_half_shape = Some(shape);
    }

    fn set_right_half_shape(&mut self, shape: NeuralNetworkShape) {
        self.right_half_shape = Some(shape);
    }

    fn reset_half_shapes(&mut self) {
        self.left_half_shape = None;
        self.right_half_shape = None;
    }

    pub fn allocate(&mut self) {
        self.nn.lock().unwrap().allocate();
    }
}

impl Phenotype for NeuralNetworkPhenotype {
    fn crossover(&mut self, other: &Self) {
        let left_original_nn = self.get_nn();
        let right_original_nn = other.get_nn();
        let left_index_begin = 0;
        let mut left_index_end = left_original_nn.shape().num_layers() / 2;
        if left_index_end == 0 {
            left_index_end = 1;
        }
        let right_index_begin = right_original_nn.shape().num_layers() / 2;
        let mut right_index_end = right_original_nn.shape().num_layers();
        if right_index_end == right_index_begin {
            right_index_end += 1;
        }
        let left_half_shape = left_original_nn
            .shape()
            .clone()
            .cut_out(left_index_begin, left_index_end);
        let right_half_shape = right_original_nn
            .shape()
            .clone()
            .cut_out(right_index_begin, right_index_end);
        self.set_left_half_shape(left_half_shape);
        self.set_right_half_shape(right_half_shape);
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        let mut rng_wrapper = RealRng::new(rng);
        let left_half_shape = self.left_half_shape.clone();
        let right_half_shape = self.right_half_shape.clone();
        let previous_shape = if left_half_shape.is_some() && right_half_shape.is_some() {
            let left_shape = left_half_shape.unwrap();
            let right_shape = right_half_shape.unwrap();
            left_shape.merge(right_shape, fetch_activation_data(&mut rng_wrapper))
        } else {
            self.get_nn().shape().clone()
        };

        let mut mutater = NeuralNetworkMutater::new(&mut rng_wrapper);

        let mut mutated_shape = mutater.mutate_shape(previous_shape.clone());
        let mut i = 0;
        while mutated_shape.to_neural_network_shape() == previous_shape {
            mutated_shape = mutater.mutate_shape(previous_shape.clone());
            i += 1;
            if i > 10 {
                break;
            }
        }
        let nn = TrainableClassicNeuralNetwork::new(
            mutated_shape.to_neural_network_shape(),
            self.nn.lock().unwrap().get_model_directory(),
        );
        self.set_nn(nn);
        self.reset_half_shapes();
    }
}

impl Adjust<NeuralNetworkPhenotype> for NeuralNetworkPhenotype {
    fn incr_number_mutates(&mut self) -> usize {
        self.nb_mutates += 1;
        self.nb_mutates
    }

    fn decr_number_mutates(&mut self) -> usize {
        if self.nb_mutates > 0 {
            self.nb_mutates -= 1;
        }
        self.nb_mutates
    }

    fn get_number_mutates(&self) -> usize {
        self.nb_mutates
    }
}
