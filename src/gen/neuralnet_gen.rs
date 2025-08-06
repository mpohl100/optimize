use crate::neural::nn::nn_factory::new_trainable_neural_network;
use crate::neural::nn::nn_factory::trainable_neural_network_from_disk;
use crate::neural::nn::nn_factory::NeuralNetworkCreationArguments;
use crate::neural::nn::nn_trait::WrappedTrainableNeuralNetwork;
use crate::neural::training::data_importer::DataImporter;

use crate::evol::evolution::EvolutionOptions;
use crate::evol::evolution::ParallelEvolutionLauncher;
use crate::evol::rng::RandomNumberGenerator;

use crate::gen::challenge::nn_challenge::NeuralNetworkChallenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::neural::training::training_params::TrainingParams;
use crate::neural::utilities::util::WrappedUtils;

use super::strategy::nn_strategy::NeuralNetworkStrategy;

pub struct NeuralNetworkGenerator {
    num_threads: usize,
    params: TrainingParams,
    evolution_params: EvolutionOptions,
    current_winner: WrappedTrainableNeuralNetwork,
    data_importer: Box<dyn DataImporter + Send + Sync>,
}

impl NeuralNetworkGenerator {
    pub fn new(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send + Sync>,
        model_directory: String,
        num_threads: usize,
        utils: WrappedUtils,
    ) -> Self {
        let nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
            params.shape().clone(),
            params.levels(),
            params.pre_shape(),
            model_directory,
            utils.clone(),
        ));
        Self { current_winner: nn, params, evolution_params, num_threads, data_importer }
    }

    pub fn from_disk(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send + Sync>,
        model_directory: String,
        num_threads: usize,
        utils: WrappedUtils,
    ) -> Self {
        let nn = trainable_neural_network_from_disk(model_directory.clone(), utils);
        let mut changed_params = params.clone();
        changed_params.set_shape(nn.shape().clone());
        Self {
            current_winner: nn,
            params: changed_params,
            evolution_params,
            num_threads,
            data_importer,
        }
    }

    /// Generate a new neural network using a genetic algorithm
    pub fn generate(&mut self) {
        let mut rng = RandomNumberGenerator::new();

        assert!(self.current_winner.shape().is_valid());
        assert!(self.current_winner.shape().num_layers() > 0);
        // make sure both shapes are the same
        self.params.set_shape(self.current_winner.shape().clone());

        let starting_value = NeuralNetworkPhenotype::new(self.current_winner.clone());
        let options = self.evolution_params.clone();
        let challenge =
            NeuralNetworkChallenge::new(self.params.clone(), self.data_importer.clone());
        let strategy =
            NeuralNetworkStrategy::new(self.current_winner.get_model_directory().path().clone());
        let launcher: ParallelEvolutionLauncher<
            NeuralNetworkPhenotype,
            NeuralNetworkStrategy,
            NeuralNetworkChallenge,
        > = ParallelEvolutionLauncher::new(strategy.clone(), challenge.clone(), self.num_threads);
        let result = launcher.evolve(&options, starting_value, &mut rng);
        self.current_winner = result.unwrap().pheno.get_nn().clone();
    }

    /// Save the current winner to disk
    pub fn save(&mut self) {
        let _ = self.current_winner.save(self.current_winner.get_model_directory().path().clone());
    }

    /// Get the model directory
    pub fn get_model_directory(&self) -> String {
        self.current_winner.get_model_directory().path().clone()
    }
}
