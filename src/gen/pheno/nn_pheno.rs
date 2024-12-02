use super::nn_mutater::NeuralNetworkMutater;
use crate::evol::phenotype::Phenotype;
use crate::evol::rng::RandomNumberGenerator;
use crate::neural::nn::neuralnet::NeuralNetwork;
use crate::neural::nn::shape::NeuralNetworkShape;

use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct NeuralNetworkPhenotype {
    nn_shape: NeuralNetworkShape,
    nn: Arc<Mutex<NeuralNetwork>>,
}

impl Clone for NeuralNetworkPhenotype {
    fn clone(&self) -> Self {
        Self {
            nn_shape: self.nn_shape.clone(),
            nn: Arc::new(Mutex::new(self.get_nn())),
        }
    }
}

impl NeuralNetworkPhenotype {
    pub fn new(nn: NeuralNetwork) -> Self {
        let nn_shape = nn.shape();
        Self {
            nn_shape: nn_shape.clone(),
            nn: Arc::new(Mutex::new(nn)),
        }
    }

    pub fn get_nn(&self) -> NeuralNetwork {
        self.nn.lock().unwrap().clone()
    }

    pub fn set_nn(&mut self, nn: NeuralNetwork) {
        *self.nn.lock().unwrap() = nn;
        self.deduce_shape();
    }

    fn deduce_shape(&mut self) {
        self.nn_shape = self.get_nn().shape().clone();
    }
}

impl Phenotype for NeuralNetworkPhenotype {
    fn crossover(&mut self, other: &Self) {
        let left_half_shape = self
            .nn_shape
            .clone()
            .cut_out(0, self.nn_shape.num_layers() / 2);
        let right_half_shape = other.nn_shape.clone().cut_out(
            self.nn_shape.num_layers() / 2 + 1,
            self.nn_shape.num_layers(),
        );
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
        self.nn_shape = self.nn.lock().unwrap().shape().clone();
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        let mut mutater = NeuralNetworkMutater::new(rng);
        let mutated_shape = mutater.mutate_shape(self.nn_shape.clone());
        self.nn.lock().unwrap().adapt_to_shape(mutated_shape);
    }
}
