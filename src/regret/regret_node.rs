use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct RegretNode {
    probability: f64,
    min_probability: f64,
    current_expected_value: f64,
    parent: Option<WrappedRegretNode>,
    children: Vec<WrappedRegretNode>,
}

impl RegretNode {
    pub fn new(probability: f64, min_probability: f64, parent: Option<WrappedRegretNode>) -> Self {
        RegretNode {
            probability,
            min_probability,
            current_expected_value: 0.0,
            parent,
            children: Vec::new(),
        }
    }

    pub fn solve(&mut self, num_iterations: usize) {
        for _ in 0..num_iterations {
            // Implement the regret minimization algorithm here
            self.calculate_expected_value();
            self.calculate_regrets(self.current_expected_value);
            let sumRegrets = self.get_sum_regrets();
            self.calculate_probabilities(sumRegrets);
            self.calculate_normalized_probabilities();
            self.update_average_values();
        }
    }

    fn calculate_expected_value(&mut self) -> f64 {
        // Calculate the expected value based on the current probabilities
        if self.get_total_probability() < self.min_probability {
            self.current_expected_value = 0.0;
            return 0.0;
        }

        self.populate_children();

        self.current_expected_value = 0.0;
        for child in self.get_children() {
            self.current_expected_value += child.get_expected_value() * child.get_probability();
        }
        self.current_expected_value
    }

    fn get_total_probability(&self) -> f64 {
        let parent_probability = self.parent.as_ref().map_or(1.0, |p| p.get_total_probability());
        parent_probability * self.probability
    }

    fn get_children(&self) -> Vec<WrappedRegretNode> {
        self.children.clone()
    }

    fn populate_children(&mut self){
        unimplemented!();
    }

    fn get_expected_value(&self) -> f64 {
        self.current_expected_value
    }
}

#[derive(Clone)]
struct WrappedRegretNode {
    node: Arc<Mutex<RegretNode>>,
}   

impl WrappedRegretNode {
    pub fn new(node: RegretNode) -> Self {
        WrappedRegretNode {
            node: Arc::new(Mutex::new(node)),
        }
    }

    pub fn get_total_probability(&self) -> f64 {
        self.node.lock().unwrap().get_total_probability()
    }

    pub fn get_expected_value(&self) -> f64 {
        self.node.lock().unwrap().get_expected_value()
    }
}
