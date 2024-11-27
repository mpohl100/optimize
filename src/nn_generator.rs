use learn::evol::evolution::EvolutionOptions;
use learn::evol::evolution::LogLevel;
use learn::gen::neuralnet_gen::NeuralNetworkGenerator;
use learn::neural::nn::shape::NeuralNetworkShape;
use learn::neural::training::data_importer::{DataImporter, SessionData};

use clap::Parser;
use learn::neural::training::training_params::TrainingParams;

/// Command line arguments
#[derive(Parser)]
struct Args {
    /// Directory where the model shall be saved
    #[clap(long)]
    model_directory: String,
    #[clap(long)]
    shape_file: String,

    // insert the training params here
    #[clap(long, default_value = "700")]
    num_training_samples: usize,
    #[clap(long, default_value = "300")]
    num_verification_samples: usize,
    #[clap(long, default_value = "0.01")]
    learning_rate: f64,
    #[clap(long, default_value = "100")]
    epochs: usize,
    #[clap(long, default_value = "0.1")]
    tolerance: f64,

    // insert the evolution options here
    #[clap(long, default_value = "100")]
    num_generations: usize,
    #[clap(long, default_value = "1")]
    log_level: usize,
    #[clap(long, default_value = "4")]
    population_size: usize,
    #[clap(long, default_value = "10")]
    num_offsprings: usize,

    // insert data importer params here
    #[clap(long)]
    input_file: String,
    #[clap(long)]
    target_file: String,
}

impl Args {
    fn get_training_params(&self) -> TrainingParams {
        let shape = NeuralNetworkShape::from_file(self.shape_file.clone());
        TrainingParams::new(
            shape,
            self.num_training_samples,
            self.num_verification_samples,
            self.learning_rate,
            self.epochs,
            self.tolerance,
        )
    }

    fn get_evolution_options(&self) -> EvolutionOptions {
        EvolutionOptions::new(
            self.num_generations,
            match self.log_level {
                0 => LogLevel::None,
                1 => LogLevel::Minimal,
                2 => LogLevel::Verbose,
                _ => LogLevel::None,
            },
            self.population_size,
            self.num_offsprings,
        )
    }
}

// Mock DataImporter implementation for testing
#[derive(Clone)]
struct MockDataImporter {
    shape: NeuralNetworkShape,
}

impl MockDataImporter {
    fn new(shape: NeuralNetworkShape) -> Self {
        Self { shape }
    }
}

impl DataImporter for MockDataImporter {
    fn get_data(&self) -> SessionData {
        let num_samples = 1000;
        let input_size = self.shape.layers[0].input_size(); // Input dimension (e.g., 28x28 image flattened)
        let num_classes = self.shape.layers[self.shape.layers.len() - 1].output_size(); // Number of output classes (e.g., for digit classification)

        // Initialize inputs and targets with zeros
        let data = vec![vec![0.0; input_size]; num_samples];
        let labels = vec![vec![0.0; num_classes]; num_samples];

        SessionData { data, labels }
    }
}

fn main() {
    let args = Args::parse();
    let model_directory = &args.model_directory;

    let training_params = args.get_training_params();

    let evolution_options = args.get_evolution_options();

    let data_importer = MockDataImporter::new(training_params.shape().clone());

    let mut nn_generator = NeuralNetworkGenerator::new(
        training_params,
        evolution_options,
        Box::new(data_importer),
        model_directory.clone(),
    );
    nn_generator.generate();
    nn_generator.save();
}
