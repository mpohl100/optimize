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
struct FileDataImporter {
    shape: NeuralNetworkShape,
    input_file: String,
    target_file: String,
}

impl FileDataImporter {
    fn new(shape: NeuralNetworkShape, input_file: String, target_file: String) -> Self {
        Self {
            shape,
            input_file,
            target_file,
        }
    }
}

impl DataImporter for FileDataImporter {
    fn get_data(&self) -> SessionData {
        let input_size = self.shape.layers[0].input_size(); // Input dimension (e.g., 28x28 image flattened)
        let num_classes = self.shape.layers[self.shape.layers.len() - 1].output_size(); // Number of output classes (e.g., for digit classification)

        // Initialize inputs and targets with zeros
        // read data from input csv file
        let data = self.read_data(self.input_file.clone());
        let labels = self.read_data(self.target_file.clone());

        // check that sizes match
        assert_eq!(data.len(), labels.len());
        for i in 0..data.len() {
            assert_eq!(data[i].len(), input_size);
            assert_eq!(labels[i].len(), num_classes);
        }

        SessionData { data, labels }
    }
}

impl FileDataImporter {
    fn read_data(&self, file: String) -> Vec<Vec<f64>> {
        let mut rdr = csv::Reader::from_path(file).unwrap();
        let mut data = Vec::new();
        for result in rdr.records() {
            let record = result.unwrap();
            let mut row = Vec::new();
            for value in record.iter() {
                row.push(value.parse::<f64>().unwrap());
            }
            data.push(row);
        }
        data
    }
}

fn main() {
    let args = Args::parse();
    let model_directory = &args.model_directory;

    let training_params = args.get_training_params();

    let evolution_options = args.get_evolution_options();

    let data_importer = FileDataImporter::new(
        training_params.shape().clone(),
        args.input_file,
        args.target_file,
    );

    let mut nn_generator = NeuralNetworkGenerator::new(
        training_params,
        evolution_options,
        Box::new(data_importer),
        model_directory.clone(),
    );
    nn_generator.generate();
    nn_generator.save();
}
