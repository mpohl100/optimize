use crate::evol::evolution::EvolutionOptions;
use crate::evol::rng::RandomNumberGenerator;
use crate::evol::strategy::BreedStrategy;
use crate::evol::strategy::OrdinaryStrategy;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;

use std::fmt::Error;

#[derive(Debug, Clone)]
pub struct NeuralNetworkStrategy {
    model_directory: String,
}

impl NeuralNetworkStrategy {
    pub fn new(model_directory: String) -> Self {
        Self { model_directory }
    }
}

impl BreedStrategy<NeuralNetworkPhenotype> for NeuralNetworkStrategy {
    fn breed(
        &self,
        parents: &[NeuralNetworkPhenotype],
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<NeuralNetworkPhenotype>, Error> {
        let ordinary_strategy = OrdinaryStrategy;
        let _ = parents[0].get_nn().save(self.model_directory.clone());
        ordinary_strategy.breed(parents, evol_options, rng)
    }
}
