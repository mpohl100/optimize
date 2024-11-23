use crate::evol::evolution::challenge::Challenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::neural::training::data_importer::DataImporter;
use crate::neural::training::training_session::TrainingSession;
use crate::neural::training::training_params::TrainingParams;

pub struct NeuralNetworkChallenge{
    params: TrainingParams,
    data_importer: Box<dyn DataImporter>,
}

impl NeuralNetworkChallenge {
    pub fn new(params: TrainingParams, data_importer: Box<dyn DataImporter>) -> Self {
        Self {
            params: params,
            data_importer: data_importer,
        }
    }
}

impl Challenge<NeuralNetworkPhenotype> for NeuralNetworkChallenge{
    fn score(&self, phenotype: &NeuralNetworkPhenotype) -> f64 {
        let mut training_session = TrainingSession::from_network(phenotype.get_nn(),self.params.clone(),self.data_importer.clone()).unwrap();
        let result = training_session.train();
        result.unwrap()
    }
} 