use crate::neural::nn::directory::Directory;
use crate::neural::nn::neuralnet::TrainableClassicNeuralNetwork;
use crate::neural::training::data_importer::DataImporter;

use crate::evol::evolution::EvolutionOptions;
use crate::evol::evolution::ParallelEvolutionLauncher;
use crate::evol::rng::RandomNumberGenerator;

use crate::gen::challenge::nn_challenge::NeuralNetworkChallenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::neural::nn::nn_trait::NeuralNetwork;
use crate::neural::training::training_params::TrainingParams;

use super::strategy::nn_strategy::NeuralNetworkStrategy;

pub struct NeuralNetworkGenerator {
    model_directory: String,
    nb_threads: usize,
    params: TrainingParams,
    evolution_params: EvolutionOptions,
    current_winner: TrainableClassicNeuralNetwork,
    data_importer: Box<dyn DataImporter + Send + Sync>,
}

impl NeuralNetworkGenerator {
    pub fn new(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send + Sync>,
        model_directory: String,
        nb_threads: usize,
    ) -> Self {
        let nn = TrainableClassicNeuralNetwork::new(
            params.shape().clone(),
            Directory::User(model_directory.clone()),
        );
        let dir = nn.get_model_directory();
        Self {
            current_winner: nn,
            params,
            evolution_params,
            model_directory: dir.path(),
            nb_threads,
            data_importer,
        }
    }

    pub fn from_disk(
        params: TrainingParams,
        evolution_params: EvolutionOptions,
        data_importer: Box<dyn DataImporter + Send + Sync>,
        model_directory: String,
        nb_threads: usize,
    ) -> Self {
        let nn = TrainableClassicNeuralNetwork::from_disk(model_directory.clone());
        if nn.is_some() {
            let mut changed_params = params.clone();
            changed_params.set_shape(nn.as_ref().unwrap().shape().clone());
            Self {
                current_winner: nn.unwrap(),
                params: changed_params,
                evolution_params,
                model_directory: model_directory.clone(),
                nb_threads,
                data_importer,
            }
        } else {
            Self::new(
                params,
                evolution_params,
                data_importer,
                model_directory.clone(),
                nb_threads,
            )
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
        let strategy = NeuralNetworkStrategy::new(self.model_directory.clone());
        let launcher: ParallelEvolutionLauncher<
            NeuralNetworkPhenotype,
            NeuralNetworkStrategy,
            NeuralNetworkChallenge,
        > = ParallelEvolutionLauncher::new(strategy.clone(), challenge.clone(), self.nb_threads);
        let result = launcher.evolve(&options, starting_value, &mut rng);
        self.current_winner = result.unwrap().pheno.get_nn().clone();
    }

    /// Save the current winner to disk
    pub fn save(&mut self) {
        let _ = self.current_winner.save(self.model_directory.clone());
    }
}
