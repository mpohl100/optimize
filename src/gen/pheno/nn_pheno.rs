use super::{nn_mutater::NeuralNetworkMutater, rng_wrapper::RealRng};
use crate::evol::phenotype::Phenotype;
use crate::evol::rng::RandomNumberGenerator;
use crate::neural::nn::neuralnet::NeuralNetwork;

use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct NeuralNetworkPhenotype {
    nn: Arc<Mutex<NeuralNetwork>>,
}

impl Clone for NeuralNetworkPhenotype {
    fn clone(&self) -> Self {
        Self {
            nn: Arc::new(Mutex::new(self.get_nn())),
        }
    }
}

impl NeuralNetworkPhenotype {
    pub fn new(nn: NeuralNetwork) -> Self {
        Self {
            nn: Arc::new(Mutex::new(nn)),
        }
    }

    pub fn get_nn(&self) -> NeuralNetwork {
        self.nn.lock().unwrap().clone()
    }

    pub fn set_nn(&mut self, nn: NeuralNetwork) {
        *self.nn.lock().unwrap() = nn;
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
        let left_nn = self
            .get_nn()
            .get_subnetwork(left_half_shape)
            .expect("Failed to get left subnetwork");
        let right_nn = other
            .get_nn()
            .get_subnetwork(right_half_shape)
            .expect("Failed to get right subnetwork");
        let new_nn = left_nn.merge(right_nn);
        self.set_nn(new_nn);
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        let mut rng_wrapper = RealRng::new(rng);
        let mut mutater = NeuralNetworkMutater::new(&mut rng_wrapper);
        let previous_shape = self.get_nn().shape().clone();
        let mut mutated_shape = mutater.mutate_shape(previous_shape.clone());
        let mut i = 0;
        while mutated_shape.to_neural_network_shape() == previous_shape {
            mutated_shape = mutater.mutate_shape(previous_shape.clone());
            i += 1;
            if i > 10 {
                break;
            }
        }
        let mut nn = self.get_nn();
        nn.adapt_to_shape(mutated_shape);
        self.set_nn(nn);
    }
}
