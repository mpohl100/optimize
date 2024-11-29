use crate::evol::evolution::challenge::Challenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::neural::training::data_importer::DataImporter;
use crate::neural::training::training_params::TrainingParams;
use crate::neural::training::training_session::TrainingSession;

use std::sync::{Arc, Mutex};

pub struct NeuralNetworkChallenge {
    params: TrainingParams,
    data_importer: Arc<Mutex<Box<dyn DataImporter + Send>>>,
}

impl NeuralNetworkChallenge {
    pub fn new(params: TrainingParams, data_importer: Box<dyn DataImporter + Send>) -> Self {
        Self {
            params,
            data_importer: Arc::new(Mutex::new(data_importer)),
        }
    }
}

impl Challenge<NeuralNetworkPhenotype> for NeuralNetworkChallenge {
    fn score(&self, phenotype: &mut NeuralNetworkPhenotype) -> f64 {
        let mut training_session = TrainingSession::from_network(
            phenotype.get_nn(),
            self.params.clone(),
            self.data_importer.lock().unwrap().clone(),
        )
        .unwrap();
        let result = training_session.train();
        phenotype.set_nn(training_session.get_nn().clone());
        result.unwrap()
    }
}
