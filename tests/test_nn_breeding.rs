use learn::evol::rng::RandomNumberGenerator;
use learn::neural::nn::nn_factory::{new_trainable_neural_network, NeuralNetworkCreationArguments};
use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};

use learn::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use learn::gen::strategy::nn_strategy::NeuralNetworkStrategy;

use learn::evol::evolution::EvolutionOptions;
use learn::evol::evolution::LogLevel;
use learn::evol::strategy::BreedStrategy;
use learn::neural::utilities::util::{Utils, WrappedUtils};

#[test]
fn test_neural_network_breeding() {
    // Define the neural network shape
    let nn_shape = NeuralNetworkShape {
        layers: vec![
            LayerShape {
                layer_type: LayerType::Dense { input_size: 128, output_size: 128 },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense { input_size: 128, output_size: 64 },
                activation: ActivationData::new(ActivationType::ReLU),
            },
            LayerShape {
                layer_type: LayerType::Dense { input_size: 64, output_size: 10 },
                activation: ActivationData::new(ActivationType::Sigmoid),
            },
        ],
    };

    let input_data = vec![0.0; 128]; // Example input data

    let utils = WrappedUtils::new(Utils::new(1000000000, 4));

    // Create a neural network phenotype
    let mut nn = new_trainable_neural_network(NeuralNetworkCreationArguments::new(
        nn_shape,
        None,
        None,
        "breeding_test_model".to_string(),
        utils.clone(),
    ));
    let _ = nn.predict(input_data);
    let nn_phenotype = NeuralNetworkPhenotype::new(nn);
    let mut parents = vec![nn_phenotype];

    let evol_opts = EvolutionOptions::new(100, LogLevel::None, 4, 10);

    let mut rng = RandomNumberGenerator::new();

    let model_directory = "breeding_test_model".to_owned();

    let nn_strategy = NeuralNetworkStrategy::new(model_directory.clone());

    for _ in 0..20 {
        let children = nn_strategy.breed(&parents, &evol_opts, &mut rng).expect("Breed failed");
        assert_eq!(children.len(), 10);
        assert!(children.iter().all(|child| child.get_nn().shape().is_valid()));
        parents.clear();
        for child in children.iter().take(4) {
            parents.push(child.clone());
        }
        parents[0].allocate();
    }

    // Remove model directory
    // std::fs::remove_dir_all(model_directory).expect("Failed to remove model directory");
}
