

pub struct RegretNode {
  probability: f64,
  min_probability: f64,
}

impl RegretNode {
    pub fn new(probability: f64, min_probability: f64) -> Self {
        RegretNode {
            probability,
            min_probability,
        }
    }
}