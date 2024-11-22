use crate::neural::nn::neuralnet::NeuralNetwork;
use crate::neural::nn::shape::NeuralNetworkShape;

use crate::evol::evolution::EvolutionLauncher;
use crate::evol::evolution::EvolutionOptions;
use crate::evol::strategy::OrdinaryStrategy;
use crate::evol::rng::RandomNumberGenerator;

use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::gen::challenge::nn_challenge::NeuralNetworkChallenge;

struct NeuralNetworkGenerator{
    model_directory: String,
    current_winner: NeuralNetwork,
}

impl NeuralNetworkGenerator {
    pub fn new(shape: NeuralNetworkShape, model_directory: String) -> Self {
        let nn = NeuralNetwork::new(shape);
        Self {
            current_winner: nn,
            model_directory: model_directory,
        }
    }

    pub fn from_disk(model_directory: &String) -> Self {
        let nn = NeuralNetwork::from_disk(model_directory);
        Self {
            current_winner: nn,
            model_directory: model_directory.clone(),
        }
    }

    pub fn generate(&mut self){
        let mut rng = RandomNumberGenerator::new();
        let starting_value = NeuralNetworkPhenotype::new(self.current_winner.clone());
        let options = EvolutionOptions::default();
        let challenge = NeuralNetworkChallenge::new();
        let strategy = OrdinaryStrategy;
        let launcher: EvolutionLauncher<NeuralNetworkPhenotype, OrdinaryStrategy, NeuralNetworkChallenge> =
            EvolutionLauncher::new(strategy, challenge);
        let result = launcher.evolve(&options, starting_value, &mut rng);
        self.current_winner = result.unwrap().pheno.get_nn().clone();
        self.current_winner.save(self.model_directory.clone());
    }
}