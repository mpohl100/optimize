use super::nn_mutater::NeuralNetworkMutater;
use crate::evol::phenotype::Phenotype;
use crate::evol::rng::RandomNumberGenerator;
use crate::neural::nn::neuralnet::NeuralNetwork;
use crate::neural::nn::shape::NeuralNetworkShape;

#[derive(Debug, Clone)]
pub struct NeuralNetworkPhenotype {
    nn_shape: NeuralNetworkShape,
    nn: NeuralNetwork,
}

impl NeuralNetworkPhenotype {
    pub fn new(nn: NeuralNetwork) -> Self {
        let nn_shape = nn.shape();
        Self {
            nn_shape: nn_shape.clone(),
            nn,
        }
    }

    pub fn get_nn(&self) -> NeuralNetwork {
        self.nn.clone()
    }

    pub fn set_nn(&mut self, nn: NeuralNetwork) {
        self.nn = nn;
    }
}

impl Phenotype for NeuralNetworkPhenotype {
    fn crossover(&mut self, _other: &Self) {
        // do nothing in crossover as it is hard to guess which feature of which neural net to pick
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        let mut mutater = NeuralNetworkMutater::new(rng);
        let mutated_shape = mutater.mutate_shape(self.nn_shape.clone());
        self.nn.adapt_to_shape(mutated_shape);
    }
}
