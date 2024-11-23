use crate::neural::nn::neuralnet::NeuralNetwork;
use crate::neural::nn::shape::NeuralNetworkShape;
use crate::neural::training::data_importer::DataImporter;

use crate::evol::evolution::EvolutionLauncher;
use crate::evol::evolution::LogLevel;
use crate::evol::evolution::EvolutionOptions;
use crate::evol::strategy::OrdinaryStrategy;
use crate::evol::rng::RandomNumberGenerator;

use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::gen::challenge::nn_challenge::NeuralNetworkChallenge;

pub struct NeuralNetworkGenerator{
    model_directory: String,
    current_winner: NeuralNetwork,
    data_importer: Box<dyn DataImporter>,
}

impl NeuralNetworkGenerator {
    pub fn new(shape: NeuralNetworkShape, data_importer: Box<dyn DataImporter>, model_directory: String) -> Self {
        let nn = NeuralNetwork::new(shape);
        Self {
            current_winner: nn,
            model_directory: model_directory,
            data_importer: data_importer,
        }
    }

    pub fn from_disk(data_importer: Box<dyn DataImporter>, model_directory: &String) -> Self {
        let nn = NeuralNetwork::from_disk(model_directory);
        Self {
            current_winner: nn,
            model_directory: model_directory.clone(),
            data_importer: data_importer,
        }
    }

    /// Generate a new neural network using a genetic algorithm
    pub fn generate(&mut self){
        let mut rng = RandomNumberGenerator::new();
        let starting_value = NeuralNetworkPhenotype::new(self.current_winner.clone());
        let options = EvolutionOptions::new(1, LogLevel::Verbose, 1, 4);
        let challenge = NeuralNetworkChallenge::new(self.data_importer.clone());
        let strategy = OrdinaryStrategy;
        let launcher: EvolutionLauncher<NeuralNetworkPhenotype, OrdinaryStrategy, NeuralNetworkChallenge> =
            EvolutionLauncher::new(strategy, challenge);
        let result = launcher.evolve(&options, starting_value, &mut rng);
        self.current_winner = result.unwrap().pheno.get_nn().clone();
    }

    /// Save the current winner to disk
    pub fn save(&self){
        let _ = self.current_winner.save(self.model_directory.clone());
    }
}