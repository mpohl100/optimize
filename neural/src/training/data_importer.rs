use dyn_clone::DynClone;

pub struct SessionData {
    pub data: Vec<Vec<f64>>,
    pub labels: Vec<Vec<f64>>,
}

pub trait DataImporter: DynClone {
    fn get_data(&self) -> SessionData;
}

dyn_clone::clone_trait_object!(DataImporter);
