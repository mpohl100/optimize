use neural::nn::shape::NeuralNetworkShape;
use neural::nn::shape::{ActivationData, ActivationType, LayerShape, LayerType};
use neural::training::data_importer::{DataImporter, SessionData};
use neural::training::training_params::TrainingParams;
use neural::utilities::util::{Utils, WrappedUtils};
use solver::neural_solver::NeuralSolver;

use clap::Parser;
use neural::layer::dense_layer::MatrixParams;

/// Command line arguments
#[derive(Parser)]
struct Args {
    /// Directory where the model shall be saved
    #[clap(long)]
    model_directory: String,
    #[clap(long, default_value = "")]
    shape_file: String,
    #[clap(long, default_value = "0")]
    retry_levels: i32,
    #[clap(long, default_value = "")]
    pre_shape: String,
    #[clap(long, default_value = "4")]
    num_threads: usize,
    #[clap(long, default_value = "1000000000")]
    cpu_memory: usize,

    // training params
    #[clap(long, default_value = "0.7")]
    validation_split: f64,
    #[clap(long, default_value = "0.01")]
    learning_rate: f64,
    #[clap(long, default_value = "100")]
    epochs: usize,
    #[clap(long, default_value = "0.1")]
    tolerance: f64,
    #[clap(long, default_value = "32")]
    batch_size: usize,
    #[clap(long, default_value = "false")]
    use_adam: bool,
    #[clap(long, default_value = "1.0")]
    sample_match_percentage: f64,

    // data importer params
    #[clap(long)]
    input_file: String,
    #[clap(long)]
    target_file: String,

    // solver params
    #[clap(long, default_value = "10")]
    num_iterations: usize,
    #[clap(long, default_value = "false")]
    do_randomize_children: bool,
}

impl Args {
    fn get_training_params(
        &self,
        data: &SessionData,
    ) -> TrainingParams {
        let levels = if self.retry_levels > 0 { Some(self.retry_levels) } else { None };
        let pre_shape = if self.pre_shape.is_empty() {
            None
        } else {
            Some(NeuralNetworkShape::from_file(self.pre_shape.clone()))
        };
        if self.shape_file.is_empty() {
            let input_size = data.data[0].len();
            let output_size = data.labels[0].len();
            let shape = NeuralNetworkShape::new(vec![LayerShape {
                layer_type: LayerType::Dense {
                    input_size,
                    output_size,
                    matrix_params: MatrixParams { slice_rows: 1000, slice_cols: 1000 },
                },
                activation: ActivationData::new(ActivationType::Sigmoid),
            }]);
            return TrainingParams::new(
                shape,
                levels,
                pre_shape,
                self.validation_split,
                self.learning_rate,
                self.epochs,
                self.tolerance,
                self.batch_size,
                self.use_adam,
                self.sample_match_percentage,
            );
        }
        let shape = NeuralNetworkShape::from_file(self.shape_file.clone());
        let input_size = data.data[0].len();
        let output_size = data.labels[0].len();
        assert_eq!(shape.layers[0].input_size(), input_size);
        assert_eq!(shape.layers[shape.layers.len() - 1].output_size(), output_size);
        TrainingParams::new(
            shape,
            levels,
            pre_shape,
            self.validation_split,
            self.learning_rate,
            self.epochs,
            self.tolerance,
            self.batch_size,
            self.use_adam,
            self.sample_match_percentage,
        )
    }
}

#[derive(Clone)]
struct FileDataImporter {
    input_file: String,
    target_file: String,
}

impl FileDataImporter {
    fn new(
        input_file: String,
        target_file: String,
    ) -> Self {
        Self { input_file, target_file }
    }

    fn read_data(
        &self,
        file: String,
    ) -> Vec<Vec<f64>> {
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

impl DataImporter for FileDataImporter {
    fn get_data(&self) -> SessionData {
        let data = self.read_data(self.input_file.clone());
        let labels = self.read_data(self.target_file.clone());
        SessionData { data, labels }
    }
}

fn main() {
    let args = Args::parse();
    let model_directory = args.model_directory.clone();

    let utils = WrappedUtils::new(Utils::new(args.cpu_memory, args.num_threads));

    let data_importer = FileDataImporter::new(args.input_file.clone(), args.target_file.clone());
    let session_data = data_importer.get_data();

    let training_params = args.get_training_params(&session_data);

    let all_inputs = session_data.data;
    let all_targets = session_data.labels;

    let shape = training_params.shape().clone();

    let mut solver = NeuralSolver::new(shape, training_params, all_inputs, all_targets, utils);

    let result = solver.solve(args.num_iterations, args.do_randomize_children);

    match result {
        Some(mut nn) => {
            nn.save(model_directory).expect("Failed to save model");
            println!("Model saved successfully.");
        },
        None => {
            eprintln!("Solver did not produce a result.");
            std::process::exit(1);
        },
    }
}
