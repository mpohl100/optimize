use learn::neural::nn::neuralnet::NeuralNetwork;
use learn::neural::training::data_importer::{DataImporter, SessionData};

use clap::Parser;

/// Command line arguments
#[derive(Parser)]
struct Args {
    /// Directory where the model shall be saved
    #[clap(long)]
    model_directory: String,

    // insert the training params here
    #[clap(long, default_value = "0.1")]
    tolerance: f64,

    // insert data importer params here
    #[clap(long)]
    input_file: String,
    #[clap(long)]
    target_file: String,
}

// Mock DataImporter implementation for testing
#[derive(Clone)]
struct FileDataImporter {
    input_file: String,
    target_file: String,
}

impl FileDataImporter {
    fn new(input_file: String, target_file: String) -> Self {
        Self {
            input_file,
            target_file,
        }
    }
}

impl DataImporter for FileDataImporter {
    fn get_data(&self) -> SessionData {
        // Initialize inputs and targets with zeros
        // read data from input csv file
        let data = self.read_data(self.input_file.clone());
        let labels = self.read_data(self.target_file.clone());

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

    let data_importer = FileDataImporter::new(args.input_file, args.target_file);

    let nn_opt = NeuralNetwork::from_disk(model_directory);
    if nn_opt.is_none() {
        panic!("Model directory does not exist");
    }

    let mut nn = nn_opt.unwrap();

    let data = data_importer.get_data();
    let inputs = data.data;
    let targets = data.labels;

    let mut success_count = 0.0;
    for (input, target) in inputs.iter().zip(targets) {
        let output = nn.predict(input.to_vec());

        // Check if the output matches the target
        let mut nb_correct_outputs = 0;
        for (o, t) in output.iter().zip(target.iter()) {
            if (o - t).abs() < args.tolerance {
                nb_correct_outputs += 1;
            }
        }
        success_count += nb_correct_outputs as f64 / target.len() as f64;
    }

    // Return the accuracy as the fraction of successful predictions
    let accuracy = success_count / inputs.len() as f64;
    println!("Accuracy: {}", accuracy);
}
