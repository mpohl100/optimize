use learn::neural::nn::neural_network::NeuralNetwork;
use learn::neural::nn::shape::NeuralNetworkShape;

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

    pub fn from_disk(model_directory: String) -> Self {
        let nn = NeuralNetwork::from_disk(model_directory.clone());
        Self {
            current_winner: nn,
            model_directory: model_directory,
        }
    }

    pub fn generate(&self){
        let mut rng = RandomNumberGenerator::new();
        let starting_value = NeuralNetworkPhenotype::new(self.current_winner.clone());
        let options = EvolutionOptions::default();
        let challenge = NeuralNetworkChallenge::new();
        let strategy = OrdinaryStrategy;
        let launcher: EvolutionLauncher<NeuralNetworkPhenotype, OrdinaryStrategy, NeuralNetworkChallenge> =
            EvolutionLauncher::new(strategy, challenge);
        let result = launcher.evolve(starting_value, &options, &mut rng);
        self.current_winner = result.unwrap().get_phenotype().get_nn();
    }
}