use neural::nn::nn_factory::new_trainable_neural_network;
use neural::nn::nn_factory::trainable_neural_network_from_disk;
use neural::nn::nn_factory::NeuralNetworkCreationArguments;
use neural::nn::nn_trait::WrappedTrainableNeuralNetwork;
use neural::training::data_importer::DataImporter;
use neural::training::training_params::TrainingParams;
use neural::utilities::util::WrappedUtils;

use evol::evolution::EvolutionOptions;
use evol::evolution::ParallelEvolutionLauncher;
use evol::rng::RandomNumberGenerator;

use crate::challenge::nn_challenge::NeuralNetworkChallenge;
use crate::pheno::nn_pheno::NeuralNetworkPhenotype;

use super::strategy::nn_strategy::NeuralNetworkStrategy;

pub struct NeuralNetworkGenerator {
    num_threads: usize,
    params: TrainingParams,
    evolution_params: EvolutionOptions,
    current_winner: WrappedTrainableNeuralNetwork,
    data_importer: Box<dyn DataImporter + Send + Sync>,
}

impl NeuralNetworkGenerator {
    #[must_use]
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
            utils,
        ));
        Self { current_winner: nn, params, evolution_params, num_threads, data_importer }
    }

    #[must_use]
    pub fn from_disk(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send + Sync>,
        model_directory: String,
        num_threads: usize,
        utils: WrappedUtils,
    ) -> Self {
        let nn = trainable_neural_network_from_disk(model_directory, utils);
        let mut changed_params = params;
        changed_params.set_shape(nn.shape());
        Self {
            current_winner: nn,
            params: changed_params,
            evolution_params,
            num_threads,
            data_importer,
        }
    }

    /// Generate a new neural network using a genetic algorithm
    ///
    /// # Panics
    /// Panics if the current winner's shape is not valid or has zero layers.
    pub fn generate(&mut self) {
        let mut rng = RandomNumberGenerator::new();

        assert!(self.current_winner.shape().is_valid());
        assert!(self.current_winner.shape().num_layers() > 0);
        // make sure both shapes are the same
        self.params.set_shape(self.current_winner.shape());

        let starting_value = NeuralNetworkPhenotype::new(&self.current_winner);
        let options = self.evolution_params.clone();
        let challenge =
            NeuralNetworkChallenge::new(self.params.clone(), self.data_importer.clone());
        let strategy = NeuralNetworkStrategy::new(self.current_winner.get_model_directory().path());
        let launcher: ParallelEvolutionLauncher<
            NeuralNetworkPhenotype,
            NeuralNetworkStrategy,
            NeuralNetworkChallenge,
        > = ParallelEvolutionLauncher::new(strategy, challenge, self.num_threads);
        let result = launcher.evolve(&options, starting_value, &mut rng);
        self.current_winner = result.unwrap().pheno.get_nn();
    }

    /// Save the current winner to disk
    pub fn save(&mut self) {
        let _ = self.current_winner.save(self.current_winner.get_model_directory().path());
    }

    /// Get the model directory
    #[must_use]
    pub fn get_model_directory(&self) -> String {
        self.current_winner.get_model_directory().path()
    }
}
