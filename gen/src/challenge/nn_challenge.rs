use crate::pheno::nn_pheno::NeuralNetworkPhenotype;
use evol::evolution::challenge::Challenge;
use neural::training::data_importer::DataImporter;
use neural::training::training_params::TrainingParams;
use neural::training::training_session::TrainingSession;

#[derive(Clone)]
pub struct NeuralNetworkChallenge {
    params: TrainingParams,
    data_importer: Box<dyn DataImporter + Send + Sync>,
}

impl NeuralNetworkChallenge {
    #[must_use]
    pub fn new(
        params: TrainingParams,
        data_importer: Box<dyn DataImporter + Send + Sync>,
    ) -> Self {
        Self { params, data_importer }
    }
}

impl Challenge<NeuralNetworkPhenotype> for NeuralNetworkChallenge {
    fn score(
        &self,
        phenotype: &mut NeuralNetworkPhenotype,
    ) -> f64 {
        let mut training_session = TrainingSession::from_network(
            phenotype.get_nn(),
            self.params.clone(),
            self.data_importer.clone(),
        )
        .unwrap();
        let result = training_session.train();
        phenotype.set_nn(training_session.get_nn());
        result.unwrap()
    }
}
