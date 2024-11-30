use crate::neural::nn::neuralnet::NeuralNetwork;
use crate::neural::training::data_importer::DataImporter;

use crate::evol::evolution::EvolutionOptions;
use crate::evol::evolution::ParallelEvolutionLauncher;
use crate::evol::rng::RandomNumberGenerator;
use crate::evol::strategy::OrdinaryStrategy;

use crate::gen::challenge::nn_challenge::NeuralNetworkChallenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::neural::training::training_params::TrainingParams;

pub struct NeuralNetworkGenerator {
    model_directory: String,
    params: TrainingParams,
    evolution_params: EvolutionOptions,
    current_winner: NeuralNetwork,
    data_importer: Box<dyn DataImporter + Send>,
}

impl NeuralNetworkGenerator {
    pub fn new(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send>,
        model_directory: String,
    ) -> Self {
        let nn = NeuralNetwork::new(params.shape().clone());
        Self {
            current_winner: nn,
            params,
            evolution_params,
            model_directory,
            data_importer,
        }
    }

    pub fn from_disk(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send>,
        model_directory: &String,
    ) -> Self {
        let nn = NeuralNetwork::from_disk(model_directory);
        let mut changed_params = params.clone();
        changed_params.set_shape(nn.shape().clone());
        Self {
            current_winner: nn,
            params: changed_params,
            evolution_params,
            model_directory: model_directory.clone(),
            data_importer,
        }
    }

    /// Generate a new neural network using a genetic algorithm
    pub fn generate(&mut self) {
        let mut rng = RandomNumberGenerator::new();
        let starting_value = NeuralNetworkPhenotype::new(self.current_winner.clone());
        let options = self.evolution_params.clone();
        let challenge =
            NeuralNetworkChallenge::new(self.params.clone(), self.data_importer.clone());
        let strategy = OrdinaryStrategy;
        let launcher: ParallelEvolutionLauncher<
            NeuralNetworkPhenotype,
            OrdinaryStrategy,
            NeuralNetworkChallenge,
        > = ParallelEvolutionLauncher::new(strategy, challenge, 4);
        let result = launcher.evolve(&options, starting_value, &mut rng);
        self.current_winner = result.unwrap().pheno.get_nn().clone();
    }

    /// Save the current winner to disk
    pub fn save(&self) {
        let _ = self.current_winner.save(self.model_directory.clone());
    }
}
