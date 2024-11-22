use crate::evol::evolution::challenge::Challenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;

pub struct NeuralNetworkChallenge;

impl NeuralNetworkChallenge {
    pub fn new() -> Self {
        Self
    }
}

impl Challenge<NeuralNetworkPhenotype> for NeuralNetworkChallenge{
    fn score(&self, _phenotype: &NeuralNetworkPhenotype) -> f64 {
        0.0
     }
} 