pub struct SessionData {
    pub data: Vec<Vec<f64>>,
    pub labels: Vec<Vec<f64>>,
}

pub trait DataImporter {
    fn get_data(&self) -> SessionData;
}
