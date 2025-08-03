use std::sync::{Arc, Mutex};

pub trait ChildrenProvider {
    fn get_children(&self) -> Vec<WrappedRegretNode>;
}

#[derive(Clone)]
pub struct WrappedChildrenProvider {
    provider: Arc<Mutex<Box<dyn ChildrenProvider>>>,
}

impl WrappedChildrenProvider {
    pub fn new(provider: Box<dyn ChildrenProvider>) -> Self {
        WrappedChildrenProvider {
            provider: Arc::new(Mutex::new(provider)),
        }
    }

    pub fn get_children(&self) -> Vec<WrappedRegretNode> {
        self.provider.lock().unwrap().get_children()
    }
}

pub trait ExpectedValueProvider {
    fn get_expected_value(&self) -> f64;
}

#[derive(Clone)]
pub struct WrappedExpectedValueProvider {
    provider: Arc<Mutex<Box<dyn ExpectedValueProvider>>>,
}

impl WrappedExpectedValueProvider {
    pub fn new(provider: Box<dyn ExpectedValueProvider>) -> Self {
        WrappedExpectedValueProvider {
            provider: Arc::new(Mutex::new(provider)),
        }
    }

    pub fn get_expected_value(&self) -> f64 {
        self.provider.lock().unwrap().get_expected_value()
    }
}

#[derive(Clone)]
pub struct RegretNode {
    probability: f64,
    min_probability: f64,
    current_expected_value: f64,
    parent: Option<WrappedRegretNode>,
    children_provider: Option<WrappedChildrenProvider>,
    expected_value_provider: Option<WrappedExpectedValueProvider>,
    children: Vec<WrappedRegretNode>,
    regret: f64,
    sum_probabilities: f64,
    num_probabilities: f64,
    average_probability: f64,
    sum_expected_values: f64,
    num_expected_values: f64,
    average_expected_value: f64,
    is_fixed_node: bool,
}

impl RegretNode {
    pub fn new(
        probability: f64,
        min_probability: f64,
        parent: Option<WrappedRegretNode>,
        children_provider: Option<WrappedChildrenProvider>,
        expected_value_provider: Option<WrappedExpectedValueProvider>,
        is_fixed_node: bool,
    ) -> Self {
        RegretNode {
            probability,
            min_probability,
            current_expected_value: 0.0,
            parent,
            children: Vec::new(),
            children_provider,
            expected_value_provider,
            regret: 0.0,
            sum_probabilities: 0.0,
            num_probabilities: 0.0,
            average_probability: 0.0,
            sum_expected_values: 0.0,
            num_expected_values: 0.0,
            average_expected_value: 0.0,
            is_fixed_node,
        }
    }

    pub fn solve(&mut self, num_iterations: usize) {
        for _ in 0..num_iterations {
            // Implement the regret minimization algorithm here
            self.calculate_expected_value();
            self.calculate_regrets(self.current_expected_value);
            let sum_regrets = self.get_sum_regrets();
            self.calculate_probabilities(sum_regrets, self.children.len());
            self.calculate_normalized_probabilities(1.0);
            self.update_average_values();
        }
    }

    fn update_average_values(&mut self) {
        // Update the average values based on the current probabilities and expected values
        // check if probability is less than epsilon or nan and the nset to zero
        if self.probability < f64::EPSILON || self.probability.is_nan() {
            self.probability = 0.0;
        }
        let prob = self.get_probability();
        self.sum_probabilities += prob;
        self.num_probabilities += 1.0;
        self.average_probability = self.sum_probabilities / self.num_probabilities;

        if self.current_expected_value.is_nan() || self.current_expected_value < f64::EPSILON {
            self.current_expected_value = 0.0;
        }
        self.sum_expected_values += self.current_expected_value;
        self.num_expected_values += 1.0;
        self.average_expected_value = self.sum_expected_values / self.num_expected_values;
    }

    fn calculate_normalized_probabilities(&mut self, total_probability: f64) {
        if total_probability > 0.0 {
            self.probability /= total_probability;
        } else {
            self.probability = 0.0;
        }
        let total_probability_sum = self
            .children
            .iter()
            .map(|child| child.get_probability())
            .sum::<f64>();
        for child in &mut self.children {
            child.calculate_normalized_probabilities(total_probability_sum);
        }
    }

    fn calculate_probabilities(&mut self, sum_regrets: f64, total_siblings: usize) {
        if !self.is_fixed_node {
            if self.regret < 0.0 {
                self.add_probability(0.0);
            } else if sum_regrets <= 0.0 {
                if total_siblings == 0 {
                    self.add_probability(1.0);
                } else {
                    self.add_probability(1.0 / total_siblings as f64);
                }
            } else {
                let probability = self.regret / sum_regrets;
                self.add_probability(probability);
            }
        }

        let new_sum_regrets = self.get_sum_regrets();
        let total_siblings = self.children.len();
        self.children.iter_mut().for_each(|child| {
            child.calculate_probabilities(new_sum_regrets, total_siblings);
        });
    }

    fn add_probability(&mut self, probability: f64) {
        self.probability += probability;
    }

    pub fn calculate_regrets(&mut self, outer_expected_value: f64) {
        // Calculate regrets based on the expected value
        self.regret = outer_expected_value - self.current_expected_value;
        self.probability = 0.0; // Reset probability for next iteration
        self.children.iter_mut().for_each(|child| {
            child.calculate_regrets(self.current_expected_value);
        });
    }

    fn calculate_expected_value(&mut self) -> f64 {
        // Calculate the expected value based on the current probabilities
        if self.get_total_probability() < self.min_probability {
            self.current_expected_value = 0.0;
            return 0.0;
        }

        if self.expected_value_provider.is_some() {
            self.current_expected_value = self
                .expected_value_provider
                .as_ref()
                .unwrap()
                .get_expected_value();
            return self.current_expected_value;
        }

        self.populate_children();

        self.current_expected_value = 0.0;
        for child in self.get_children() {
            self.current_expected_value += child.get_expected_value() * child.get_probability();
        }
        self.current_expected_value
    }

    fn get_sum_regrets(&self) -> f64 {
        self.children
            .iter()
            .map(|child| child.node.lock().unwrap().regret)
            .sum()
    }

    fn get_total_probability(&self) -> f64 {
        let parent_probability = self
            .parent
            .as_ref()
            .map_or(1.0, |p| p.get_total_probability());
        parent_probability * self.probability
    }

    fn get_children(&self) -> Vec<WrappedRegretNode> {
        self.children.clone()
    }

    fn populate_children(&mut self) {
        self.children = if let Some(provider) = &self.children_provider {
            provider.get_children()
        } else {
            Vec::new()
        };
    }

    fn get_expected_value(&self) -> f64 {
        self.current_expected_value
    }

    pub fn get_probability(&self) -> f64 {
        self.probability
    }

    pub fn get_average_probability(&self) -> f64 {
        self.average_probability
    }

    pub fn get_average_expected_value(&self) -> f64 {
        self.average_expected_value
    }
}

#[derive(Clone)]
pub struct WrappedRegretNode {
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

    pub fn calculate_regrets(&self, outer_expected_value: f64) {
        self.node
            .lock()
            .unwrap()
            .calculate_regrets(outer_expected_value);
    }

    pub fn calculate_probabilities(&self, sum_regrets: f64, total_siblings: usize) {
        self.node
            .lock()
            .unwrap()
            .calculate_probabilities(sum_regrets, total_siblings);
    }

    pub fn get_probability(&self) -> f64 {
        self.node.lock().unwrap().get_probability()
    }

    pub fn calculate_normalized_probabilities(&self, total_probability: f64) {
        self.node
            .lock()
            .unwrap()
            .calculate_normalized_probabilities(total_probability);
    }

    pub fn get_average_probability(&self) -> f64 {
        self.node.lock().unwrap().get_average_probability()
    }

    pub fn get_average_expected_value(&self) -> f64 {
        self.node.lock().unwrap().get_average_expected_value()
    }
}
