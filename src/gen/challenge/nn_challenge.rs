use crate::evol::evolution::challenge::Challenge;
use crate::gen::pheno::nn_pheno::NeuralNetworkPhenotype;
use crate::neural::training::data_importer::DataImporter;

pub struct NeuralNetworkChallenge{
    data_importer: Box<dyn DataImporter>,
}

impl NeuralNetworkChallenge {
    pub fn new(data_importer: Box<dyn DataImporter>) -> Self {
        Self {
            data_importer: data_importer,
        }
    }
}

impl Challenge<NeuralNetworkPhenotype> for NeuralNetworkChallenge{
    fn score(&self, _phenotype: &NeuralNetworkPhenotype) -> f64 {
        0.0
     }
} 