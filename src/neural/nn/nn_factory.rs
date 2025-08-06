use std::{fs, io, path::Path};

use crate::neural::utilities::util::WrappedUtils;

use super::{
    directory::Directory,
    either_nn::{EitherNeuralNetwork, TrainableEitherNeuralNetwork},
    neuralnet::{ClassicNeuralNetwork, TrainableClassicNeuralNetwork},
    nn_trait::{WrappedNeuralNetwork, WrappedTrainableNeuralNetwork},
    retry_nn::{RetryNeuralNetwork, TrainableRetryNeuralNetwork},
    shape::NeuralNetworkShape,
};

pub struct NeuralNetworkCreationArguments {
    shape: NeuralNetworkShape,
    levels: Option<i32>,
    pre_shape: Option<NeuralNetworkShape>,
    model_directory: String,
    utils: WrappedUtils,
}

impl NeuralNetworkCreationArguments {
    pub fn new(
        shape: NeuralNetworkShape,
        levels: Option<i32>,
        pre_shape: Option<NeuralNetworkShape>,
        model_directory: String,
        utils: WrappedUtils,
    ) -> Self {
        Self { shape, levels, pre_shape, model_directory, utils }
    }
}

pub fn new_neural_network(
    neural_network_creation_arguments: NeuralNetworkCreationArguments
) -> WrappedNeuralNetwork {
    match neural_network_creation_arguments.levels {
        Some(levels) => WrappedNeuralNetwork::new(Box::new(RetryNeuralNetwork::new(
            neural_network_creation_arguments.shape,
            levels,
            neural_network_creation_arguments.model_directory,
            neural_network_creation_arguments.utils,
        ))),
        None => WrappedNeuralNetwork::new(Box::new(ClassicNeuralNetwork::new(
            neural_network_creation_arguments.shape,
            neural_network_creation_arguments.model_directory,
            neural_network_creation_arguments.utils,
        ))),
    }
}

pub fn new_trainable_neural_network(
    neural_network_creation_arguments: NeuralNetworkCreationArguments
) -> WrappedTrainableNeuralNetwork {
    match (neural_network_creation_arguments.pre_shape, neural_network_creation_arguments.levels) {
        (None, Some(levels)) => {
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableRetryNeuralNetwork::new(
                neural_network_creation_arguments.shape,
                levels,
                neural_network_creation_arguments.model_directory,
                neural_network_creation_arguments.utils,
            )))
        },
        (Some(pre_shape), Some(levels)) => {
            WrappedTrainableNeuralNetwork::new(Box::new(TrainableEitherNeuralNetwork::new(
                neural_network_creation_arguments.shape,
                pre_shape,
                levels,
                neural_network_creation_arguments.model_directory,
                neural_network_creation_arguments.utils,
            )))
        },
        _ => WrappedTrainableNeuralNetwork::new(Box::new(TrainableClassicNeuralNetwork::new(
            neural_network_creation_arguments.shape,
            Directory::Internal(neural_network_creation_arguments.model_directory),
            neural_network_creation_arguments.utils,
        ))),
    }
}

pub fn neural_network_from_disk(
    model_directory: String,
    utils: WrappedUtils,
) -> WrappedNeuralNetwork {
    // check if model directory contains a directory named primary
    if std::path::Path::new(&format!("{}/primary", model_directory)).exists() {
        return RetryNeuralNetwork::from_disk(model_directory, utils);
    }
    // check if model directory contains a directory named pre
    if std::path::Path::new(&format!("{}/pre", model_directory)).exists() {
        return EitherNeuralNetwork::from_disk(model_directory, utils);
    }
    WrappedNeuralNetwork::new(Box::new(
        ClassicNeuralNetwork::from_disk(model_directory, utils).unwrap(),
    ))
}

pub fn trainable_neural_network_from_disk(
    model_directory: String,
    utils: WrappedUtils,
) -> WrappedTrainableNeuralNetwork {
    // check if model directory contains a directory named primary
    if std::path::Path::new(&format!("{}/primary", model_directory)).exists() {
        return TrainableRetryNeuralNetwork::from_disk(model_directory, utils);
    }
    WrappedTrainableNeuralNetwork::new(Box::new(
        TrainableClassicNeuralNetwork::from_disk(model_directory, utils).unwrap(),
    ))
}

pub fn get_first_free_model_directory(model_directory: Directory) -> String {
    let model_directory_orig = model_directory.path();
    // truncate _{integer} from the end of the model_directory
    let mut model_directory = model_directory_orig.clone();
    if let Some(pos) = model_directory.rfind('_') {
        // check that the remainder is an integer
        let remainder = &model_directory[pos + 1..];
        if remainder.parse::<usize>().is_ok() {
            model_directory = model_directory[..pos].to_string();
        }
    }
    let mut i = 1;
    while std::fs::metadata(format!("{}_{}", model_directory, i)).is_ok() {
        i += 1;
    }
    model_directory = format!("{}_{}", model_directory, i);
    // create the directory to block the name
    std::fs::create_dir_all(&model_directory).unwrap();
    model_directory
}

pub fn copy_dir_recursive(
    src: &Path,
    dst: &Path,
) -> io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry_result in fs::read_dir(src)? {
        let entry = entry_result?;
        let path = entry.path();
        let dest_path = dst.join(entry.file_name());

        if path.is_dir() {
            copy_dir_recursive(&path, &dest_path)?;
        } else {
            fs::copy(&path, &dest_path)?;
        }
    }

    Ok(())
}
