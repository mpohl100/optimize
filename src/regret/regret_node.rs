use std::sync::{Arc, Mutex};

pub trait UserDataTrait: Clone {
    // Define the methods that UserData should implement
    fn get_probability(&self) -> f64;
}

pub trait ChildrenProvider<UserData: UserDataTrait> {
    fn get_children(
        &self,
        parents_data: Vec<UserData>,
    ) -> Vec<WrappedRegretNode<UserData>>;
}

#[derive(Clone)]
pub struct WrappedChildrenProvider<UserData: UserDataTrait> {
    provider: Arc<Mutex<Box<dyn ChildrenProvider<UserData>>>>,
}

impl<UserData: UserDataTrait> WrappedChildrenProvider<UserData> {
    pub fn new(provider: Box<dyn ChildrenProvider<UserData>>) -> Self {
        WrappedChildrenProvider { provider: Arc::new(Mutex::new(provider)) }
    }

    pub fn get_children(
        &self,
        parents_data: Vec<UserData>,
    ) -> Vec<WrappedRegretNode<UserData>> {
        self.provider.lock().unwrap().get_children(parents_data)
    }
}

pub trait ExpectedValueProvider<UserData: UserDataTrait> {
    fn get_expected_value(
        &self,
        parents_data: Vec<UserData>,
    ) -> f64;
}

#[derive(Clone)]
pub struct WrappedExpectedValueProvider<UserData: UserDataTrait> {
    provider: Arc<Mutex<Box<dyn ExpectedValueProvider<UserData>>>>,
}

impl<UserData: UserDataTrait> WrappedExpectedValueProvider<UserData> {
    pub fn new(provider: Box<dyn ExpectedValueProvider<UserData>>) -> Self {
        WrappedExpectedValueProvider { provider: Arc::new(Mutex::new(provider)) }
    }

    pub fn get_expected_value(
        &self,
        parents_data: Vec<UserData>,
    ) -> f64 {
        self.provider.lock().unwrap().get_expected_value(parents_data)
    }
}

#[derive(Clone)]
pub enum ProviderType<UserData: UserDataTrait> {
    Children(WrappedChildrenProvider<UserData>),
    ExpectedValue(WrappedExpectedValueProvider<UserData>),
}

#[derive(Clone)]
pub struct Provider<UserData: UserDataTrait> {
    pub provider_type: ProviderType<UserData>,
    pub user_data: Option<UserData>,
}

#[derive(Clone)]
pub struct RegretNode<UserData: UserDataTrait> {
    probability: f64,
    min_probability: f64,
    current_expected_value: f64,
    parents_data: Vec<UserData>,
    provider: Provider<UserData>,
    children: Vec<WrappedRegretNode<UserData>>,
    regret: f64,
    sum_probabilities: f64,
    num_probabilities: f64,
    average_probability: f64,
    sum_expected_values: f64,
    num_expected_values: f64,
    average_expected_value: f64,
    fixed_probability: Option<f64>,
}

impl<UserData: UserDataTrait> RegretNode<UserData> {
    pub fn new(
        probability: f64,
        min_probability: f64,
        parents_data: Vec<UserData>,
        provider: Provider<UserData>,
        fixed_probability: Option<f64>,
    ) -> Self {
        RegretNode {
            probability,
            min_probability,
            current_expected_value: 0.0,
            parents_data,
            children: Vec::new(),
            provider,
            regret: 0.0,
            sum_probabilities: 0.0,
            num_probabilities: 0.0,
            average_probability: 0.0,
            sum_expected_values: 0.0,
            num_expected_values: 0.0,
            average_expected_value: 0.0,
            fixed_probability,
        }
    }

    pub fn solve(
        &mut self,
        num_iterations: usize,
    ) {
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

    fn calculate_normalized_probabilities(
        &mut self,
        total_probability: f64,
    ) {
        if total_probability > 0.0 {
            self.probability /= total_probability;
        } else {
            self.probability = 0.0;
        }
        let total_probability_sum =
            self.children.iter().map(|child| child.get_probability()).sum::<f64>();
        for child in &mut self.children {
            child.calculate_normalized_probabilities(total_probability_sum);
        }
    }

    fn calculate_probabilities(
        &mut self,
        sum_regrets: f64,
        total_siblings: usize,
    ) {
        if self.fixed_probability.is_none() {
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
        } else {
            // If it's a fixed node, we do not calculate probabilities
            self.add_probability(self.fixed_probability.unwrap_or(0.0));
        }

        let new_sum_regrets = self.get_sum_regrets();
        let total_siblings = self.children.len();
        self.children.iter_mut().for_each(|child| {
            child.calculate_probabilities(new_sum_regrets, total_siblings);
        });
    }

    fn add_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability += probability;
    }

    pub fn calculate_regrets(
        &mut self,
        outer_expected_value: f64,
    ) {
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

        self.populate_children();

        self.current_expected_value = match self.provider.provider_type {
            ProviderType::ExpectedValue(ref provider) => {
                provider.get_expected_value(self.parents_data.clone())
            },
            ProviderType::Children(ref provider) => {
                self.children = provider.get_children(self.parents_data.clone());
                self.children
                    .iter()
                    .map(|child| child.get_expected_value() * child.get_probability())
                    .sum::<f64>()
            },
        };
        self.current_expected_value
    }

    fn get_sum_regrets(&self) -> f64 {
        self.children.iter().map(|child| child.node.lock().unwrap().regret).sum()
    }

    fn get_total_probability(&self) -> f64 {
        let parent_probability =
            self.parents_data.iter().fold(1.0, |acc, data| acc * data.get_probability());
        parent_probability * self.probability
    }

    fn get_children(&self) -> Vec<WrappedRegretNode<UserData>> {
        self.children.clone()
    }

    fn populate_children(&mut self) {
        match self.provider.provider_type {
            ProviderType::Children(ref provider) => {
                self.children = provider.get_children(self.parents_data.clone());
            },
            ProviderType::ExpectedValue(_) => {
                // If the provider is an expected value provider, we don't need to populate children
                self.children.clear();
            },
        }
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
pub struct WrappedRegretNode<UserData: UserDataTrait> {
    node: Arc<Mutex<RegretNode<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedRegretNode<UserData> {
    pub fn new(node: RegretNode<UserData>) -> Self {
        WrappedRegretNode { node: Arc::new(Mutex::new(node)) }
    }

    pub fn get_total_probability(&self) -> f64 {
        self.node.lock().unwrap().get_total_probability()
    }

    pub fn get_expected_value(&self) -> f64 {
        self.node.lock().unwrap().get_expected_value()
    }

    pub fn calculate_regrets(
        &self,
        outer_expected_value: f64,
    ) {
        self.node.lock().unwrap().calculate_regrets(outer_expected_value);
    }

    pub fn calculate_probabilities(
        &self,
        sum_regrets: f64,
        total_siblings: usize,
    ) {
        self.node.lock().unwrap().calculate_probabilities(sum_regrets, total_siblings);
    }

    pub fn get_probability(&self) -> f64 {
        self.node.lock().unwrap().get_probability()
    }

    pub fn calculate_normalized_probabilities(
        &self,
        total_probability: f64,
    ) {
        self.node.lock().unwrap().calculate_normalized_probabilities(total_probability);
    }

    pub fn get_average_probability(&self) -> f64 {
        self.node.lock().unwrap().get_average_probability()
    }

    pub fn get_average_expected_value(&self) -> f64 {
        self.node.lock().unwrap().get_average_expected_value()
    }
}

mod tests {
    use core::panic;

    use super::*;

    #[derive(Clone)]
    enum Choice {
        Rock,
        Paper,
        Scissors,
    }

    #[derive(Clone)]
    struct RoshamboData {
        choice: Choice,
    }

    impl UserDataTrait for RoshamboData {
        fn get_probability(&self) -> f64 {
            match self.choice {
                Choice::Rock => 0.33333,
                Choice::Paper => 0.33333,
                Choice::Scissors => 0.33334,
            }
        }
    }

    struct RoshamboChildrenProvider {}

    impl RoshamboChildrenProvider {
        pub fn new() -> Self {
            RoshamboChildrenProvider {}
        }
    }

    impl ChildrenProvider<RoshamboData> for RoshamboChildrenProvider {
        fn get_children(
            &self,
            parents_data: Vec<RoshamboData>,
        ) -> Vec<WrappedRegretNode<RoshamboData>> {
            if parents_data.len() < 1 {
                let mut children = Vec::new();
                for choice in [Choice::Rock, Choice::Paper, Choice::Scissors].iter() {
                    let cloned_parents_data = parents_data.clone();
                    let data = RoshamboData { choice: choice.clone() };
                    let mut parents_data = cloned_parents_data.clone();
                    parents_data.push(RoshamboData { choice: choice.clone() });
                    let node = RegretNode::new(
                        0.0,
                        0.0,
                        parents_data,
                        Provider {
                            provider_type: ProviderType::Children(WrappedChildrenProvider::new(
                                Box::new(RoshamboChildrenProvider::new()),
                            )),
                            user_data: Some(data.clone()),
                        },
                        None,
                    );
                    children.push(WrappedRegretNode::new(node));
                }
                return children;
            } else if parents_data.len() == 1 {
                let mut children = Vec::new();
                for choice in [Choice::Rock, Choice::Paper, Choice::Scissors].iter() {
                    let mut parents_data_clone = parents_data.clone();
                    let data = RoshamboData { choice: choice.clone() };
                    parents_data_clone.push(data.clone());
                    let node = RegretNode::new(
                        0.0,
                        0.0,
                        parents_data_clone,
                        Provider {
                            provider_type: ProviderType::ExpectedValue(
                                WrappedExpectedValueProvider::new(Box::new(
                                    RoshamboExpectedValueProvider::new(),
                                )),
                            ),
                            user_data: Some(data.clone()),
                        },
                        None,
                    );
                    children.push(WrappedRegretNode::new(node));
                }
                return children;
            }
            return Vec::new();
        }
    }

    struct RoshamboExpectedValueProvider {}

    impl RoshamboExpectedValueProvider {
        pub fn new() -> Self {
            RoshamboExpectedValueProvider {}
        }
    }

    impl ExpectedValueProvider<RoshamboData> for RoshamboExpectedValueProvider {
        fn get_expected_value(
            &self,
            parents_data: Vec<RoshamboData>,
        ) -> f64 {
            if parents_data.len() < 2 {
                panic!("Expected at least two parents data for expected value calculation");
            }
            let player_1_choice = &parents_data[0].choice;
            let player_2_choice = &parents_data[1].choice;
            match player_1_choice {
                Choice::Rock => match player_2_choice {
                    Choice::Rock => 0.0,     // Tie
                    Choice::Paper => -1.0,   // Paper beats Rock
                    Choice::Scissors => 1.0, // Rock beats Scissors
                },
                Choice::Paper => match player_2_choice {
                    Choice::Rock => 1.0,      // Paper beats Rock
                    Choice::Paper => 0.0,     // Tie
                    Choice::Scissors => -1.0, // Scissors beats Paper
                },
                Choice::Scissors => match player_2_choice {
                    Choice::Rock => -1.0,    // Rock beats Scissors
                    Choice::Paper => 1.0,    // Paper beats Rock
                    Choice::Scissors => 0.0, // Tie
                },
            }
        }
    }

    #[test]
    fn test_roshambo_regret_minimization() {
        let mut node = RegretNode::new(
            0.0,
            0.0,
            vec![],
            Provider {
                provider_type: ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                    RoshamboChildrenProvider {},
                ))),
                user_data: None,
            },
            Some(1.0),
        );

        node.solve(1000);
        let children = node.get_children();
        assert_eq!(children.len(), 3); // Should have three children for Rock, Paper, Scissors
        for child in children {
            let expected_value = child.get_average_expected_value();
            let probability = child.get_average_probability();
            // assert expected value is close to zero
            assert!((expected_value - 0.0).abs() < 0.01, "Expected value should be close to zero");
            // assert probability is close to 1/3
            assert!((probability - 1.0 / 3.0).abs() < 0.01, "Probability should be close to 1/3");
        }
    }
}
