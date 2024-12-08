use super::nn_mutater::NeuralNetworkMutater;
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
        let left_half_shape = left_original_nn.shape()
            .clone()
            .cut_out(0, left_original_nn.shape().num_layers() / 2);
        let right_half_shape = right_original_nn.shape()
            .clone()
            .cut_out(right_original_nn.shape().num_layers() / 2, right_original_nn.shape().num_layers());
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
        let mut mutater = NeuralNetworkMutater::new(rng);
        let mutated_shape = mutater.mutate_shape(self.get_nn().shape().clone());
        let mut nn = self.get_nn();
        nn.adapt_to_shape(mutated_shape);
        self.set_nn(nn);
    }
}
