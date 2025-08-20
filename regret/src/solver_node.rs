//! This file has been split into multiple modules:
//! - user_data.rs: UserDataTrait and WrappedUserData
//! - provider.rs: Provider traits, enums, and wrappers
//! - regret_node.rs: RegretNode and WrappedRegret
//! - roshambo.rs: Rock-Paper-Scissors example types and providers
//! - tests.rs: Unit tests
//! See lib.rs for module re-exports.

pub use crate::user_data::*;
pub use crate::provider::*;
pub use crate::regret_node::*;
pub use crate::roshambo::*;
//!
//! - [`WrappedUserData`]: Thread-safe wrapper around user data implementing `UserDataTrait`.
//! - [`WrappedChildrenProvider`]: Thread-safe wrapper for a boxed `ChildrenProvider`.
//! - [`WrappedExpectedValueProvider`]: Thread-safe wrapper for a boxed `ExpectedValueProvider`.
//! - [`WrappedProvider`]: Thread-safe wrapper for a `Provider`, which can be either a children or expected value provider.
//! - [`WrappedRegret`]: Thread-safe wrapper for a `RegretNode`.
//!
//! ## Core Structures
//!
//! - [`ProviderType`]: Enum representing either a children provider or expected value provider.
//! - [`Provider`]: Associates a provider type with optional user data.
//! - [`RegretNode`]: Represents a node in the regret minimization tree, maintaining probabilities, expected values, regrets, and children.
//!
//! ## Regret Minimization
//!
//! - Nodes can be solved using regret minimization over multiple iterations.
//! - Probabilities and expected values are updated iteratively based on regrets and provider outputs.
//! - Supports fixed probabilities for leaf nodes.
//!
//! ## Example: Rock-Paper-Scissors
//!
//! The module includes a test implementation for the Rock-Paper-Scissors game, demonstrating how to use custom user data, children providers, and expected value providers.
//!
//! ## Thread Safety
//!
//! All core data structures are wrapped in `Arc<Mutex<...>>` for safe concurrent access.
//!
//! ## Utilities
//!
//! - Uses `safe_lock` from `utils::safer` for ergonomic locking of mutexes.
//!
//! ## Usage
//!
//! Implement `UserDataTrait` for your domain-specific data, then provide custom children and expected value providers to build and solve regret trees.
//!
//! ## Tests
//!
//! Includes unit tests for children provider, expected value provider, and regret minimization in the context of Rock-Paper-Scissors.
use std::sync::{Arc, Mutex};

use utils::safer::safe_lock;

/// Trait for user-defined data in regret nodes, requiring probability management and string representation.
pub trait UserDataTrait: Default + Clone + std::fmt::Debug {
    /// Returns the probability associated with this user data.
    fn get_probability(&self) -> f64;
    /// Sets the probability associated with this user data.
    fn set_probability(
        &mut self,
        probability: f64,
    );
    /// Returns a string representation of the user data.
    fn get_data_as_string(&self) -> String;
}

/// Thread-safe wrapper around user data implementing `UserDataTrait`.
#[derive(Debug, Clone)]
pub struct WrappedUserData<UserData: UserDataTrait> {
    /// Thread-safe user data implementing `UserDataTrait`.
    user_data: Arc<Mutex<UserData>>,
}

impl<UserData: UserDataTrait> WrappedUserData<UserData> {
    /// Creates a new `WrappedUserData` containing the given user data.
    #[must_use]
    pub fn new(user_data: UserData) -> Self {
        Self { user_data: Arc::new(Mutex::new(user_data)) }
    }

    /// Returns a clone of the inner user data.
    #[must_use]
    pub fn get_user_data(&self) -> UserData {
        safe_lock(&self.user_data).clone()
    }

    /// Sets the probability of the inner user data.
    pub fn set_probability(
        &self,
        probability: f64,
    ) {
        let mut data = safe_lock(&self.user_data);
        data.set_probability(probability);
    }

    /// Gets the probability of the inner user data.
    fn get_probability(&self) -> f64 {
        safe_lock(&self.user_data).get_probability()
    }

    /// Gets the string representation of the inner user data.
    fn get_data_as_string(&self) -> String {
        safe_lock(&self.user_data).get_data_as_string()
    }
}

/// Trait for types that can generate child nodes given parent data.
pub trait ChildrenProvider<UserData: UserDataTrait>: std::fmt::Debug {
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        parents_data: Vec<WrappedUserData<UserData>>,
    ) -> Vec<WrappedRegret<UserData>>;
}

/// Thread-safe wrapper for a boxed `ChildrenProvider`.
#[derive(Debug, Clone)]
pub struct WrappedChildrenProvider<UserData: UserDataTrait> {
    /// The underlying provider, boxed and wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Box<dyn ChildrenProvider<UserData>>>>,
}

impl<UserData: UserDataTrait> WrappedChildrenProvider<UserData> {
    /// Creates a new wrapped children provider.
    #[must_use]
    pub fn new(provider: Box<dyn ChildrenProvider<UserData>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the children nodes for the given parent data.
    #[must_use]
    pub fn get_children(
        &self,
        parents_data: Vec<WrappedUserData<UserData>>,
    ) -> Vec<WrappedRegret<UserData>> {
        safe_lock(&self.provider).get_children(parents_data)
    }
}

/// Trait for types that can compute expected values given parent data.
pub trait ExpectedValueProvider<UserData: UserDataTrait>: std::fmt::Debug {
    /// Returns the expected value for the given parent data.
    fn get_expected_value(
        &self,
        parents_data: Vec<WrappedUserData<UserData>>,
    ) -> f64;
}

/// Thread-safe wrapper for a boxed `ExpectedValueProvider`.
#[derive(Debug, Clone)]
pub struct WrappedExpectedValueProvider<UserData: UserDataTrait> {
    /// The underlying provider, boxed and wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Box<dyn ExpectedValueProvider<UserData>>>>,
}

impl<UserData: UserDataTrait> WrappedExpectedValueProvider<UserData> {
    /// Creates a new wrapped expected value provider.
    #[must_use]
    pub fn new(provider: Box<dyn ExpectedValueProvider<UserData>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the expected value for the given parent data.
    #[must_use]
    pub fn get_expected_value(
        &self,
        parents_data: Vec<WrappedUserData<UserData>>,
    ) -> f64 {
        safe_lock(&self.provider).get_expected_value(parents_data)
    }
}

/// Enum representing either a children provider or expected value provider.
#[derive(Debug, Clone)]
pub enum ProviderType<UserData: UserDataTrait> {
    /// Children provider variant.
    Children(WrappedChildrenProvider<UserData>),
    /// Expected value provider variant.
    ExpectedValue(WrappedExpectedValueProvider<UserData>),
}

/// Associates a provider type with optional user data.
#[derive(Debug, Clone)]
pub struct Provider<UserData: UserDataTrait> {
    /// The provider type (children or expected value).
    pub provider_type: ProviderType<UserData>,
    /// Optional user data associated with the provider.
    pub user_data: Option<WrappedUserData<UserData>>,
}

impl<UserData: UserDataTrait> Provider<UserData> {
    /// Creates a new provider with the given type and optional user data.
    #[must_use]
    pub const fn new(
        provider_type: ProviderType<UserData>,
        user_data: Option<WrappedUserData<UserData>>,
    ) -> Self {
        Self { provider_type, user_data }
    }
}

/// Thread-safe wrapper for a `Provider`.
#[derive(Debug, Clone)]
pub struct WrappedProvider<UserData: UserDataTrait> {
    /// The underlying provider, wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Provider<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedProvider<UserData> {
    /// Creates a new wrapped provider.
    #[must_use]
    pub fn new(provider: Provider<UserData>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the provider type.
    #[must_use]
    pub fn get_provider_type(&self) -> ProviderType<UserData> {
        safe_lock(&self.provider).provider_type.clone()
    }

    /// Gets the optional user data.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedUserData<UserData>> {
        safe_lock(&self.provider).user_data.clone()
    }
}

/// Represents a node in the regret minimization tree.
#[derive(Clone)]
pub struct RegretNode<UserData: UserDataTrait> {
    /// Probability of this node.
    probability: f64,
    /// Minimum allowed probability.
    _min_probability: f64,
    /// Current expected value of this node.
    current_expected_value: f64,
    /// Data from parent nodes.
    parents_data: Vec<WrappedUserData<UserData>>,
    /// Provider for children or expected value.
    provider: WrappedProvider<UserData>,
    /// Child nodes.
    children: Vec<WrappedRegret<UserData>>,
    /// Regret value for this node.
    regret: f64,
    /// Sum of probabilities over iterations.
    sum_probabilities: f64,
    /// Number of probability updates.
    num_probabilities: f64,
    /// Average probability over iterations.
    average_probability: f64,
    /// Sum of expected values over iterations.
    sum_expected_values: f64,
    /// Number of expected value updates.
    num_expected_values: f64,
    /// Average expected value over iterations.
    average_expected_value: f64,
    /// Fixed probability for leaf nodes (if any).
    fixed_probability: Option<f64>,
}

impl<UserData: UserDataTrait> RegretNode<UserData> {
    /// Creates a new regret node.
    #[must_use]
    pub fn new(
        probability: f64,
        min_probability: f64,
        parents_data: Vec<WrappedUserData<UserData>>,
        provider: WrappedProvider<UserData>,
        fixed_probability: Option<f64>,
    ) -> Self {
        provider.get_user_data().as_mut().map_or_else(
            || {
                // If no user data is provided, we set the probability to 0.0
                UserData::set_probability(
                    &mut UserData::default(),
                    fixed_probability.unwrap_or(1.0),
                );
            },
            |data| data.set_probability(probability),
        );
        Self {
            probability,
            _min_probability: min_probability,
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

    /// Returns a string representation of the node and its children.
    #[allow(clippy::format_push_string)]
    #[must_use]
    pub fn get_data_as_string(
        &self,
        indentation: usize,
    ) -> String {
        // first put average probability and average expected value
        let mut result = format!(
            "{}Average Probability: {}\n",
            " ".repeat(indentation),
            self.average_probability
        );
        result.push_str(&format!(
            "{}Average Expected Value: {}\n",
            " ".repeat(indentation),
            self.average_expected_value
        ));
        // put the current provider type
        result.push_str(&format!(
            "{}Provider Type: {:?}\n",
            " ".repeat(indentation),
            self.provider.get_provider_type()
        ));

        // then push the provider user_data.get_data_as_string
        if let Some(data) = self.provider.get_user_data().as_ref() {
            // put indentation and newline
            result.push_str(&format!("{}{}\n", " ".repeat(indentation), data.get_data_as_string()));
        }
        // then put all children data
        // put "children" as caption
        result.push_str(&format!("{}Children:\n", " ".repeat(indentation)));
        for child in &self.children {
            result.push_str(&format!("{}\n", child.get_data_as_string(indentation + 2)));
        }
        result
    }

    /// Solves the node using regret minimization for the given number of iterations.
    pub fn solve(
        &mut self,
        num_iterations: usize,
    ) {
        self.populate_children();
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

    /// Updates the average probability and expected value statistics.
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

        // update average values of children
        for child in &mut self.children {
            child.update_average_values();
        }
    }

    /// Normalizes the probability of this node and its children.
    fn calculate_normalized_probabilities(
        &mut self,
        total_probability: f64,
    ) {
        if total_probability > 0.0 {
            self.probability /= total_probability;
        } else {
            self.probability = 0.0;
        }
        if let Some(data) = self.provider.get_user_data().as_mut() {
            data.set_probability(self.probability);
        }
        let total_probability_sum: f64 =
            self.children.iter().map(WrappedRegret::get_probability).sum();
        for child in &mut self.children {
            child.calculate_normalized_probabilities(total_probability_sum);
        }
    }

    /// Calculates probabilities for this node and its children based on regrets.
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
                    self.add_probability(
                        1.0 / f64::from(usize::try_into(total_siblings).unwrap_or(1)),
                    );
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

    /// Adds the given probability to this node.
    fn add_probability(
        &mut self,
        probability: f64,
    ) {
        self.probability += probability;
    }

    /// Calculates regrets for this node and its children.
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

    /// Calculates the expected value for this node.
    fn calculate_expected_value(&mut self) -> f64 {
        // Calculate the expected value based on the current probabilities
        // TODO fix this later
        // if self.get_total_probability() < self.min_probability {
        //     self.current_expected_value = 0.0;
        //     return 0.0;
        // }

        self.current_expected_value = match self.provider.get_provider_type() {
            ProviderType::ExpectedValue(ref provider) => {
                let mut cloned_parents_data = self.parents_data.clone();
                cloned_parents_data.extend(
                    self.provider
                        .get_user_data()
                        .as_ref()
                        .map_or_else(Vec::new, |data| vec![data.clone()]),
                );
                provider.get_expected_value(cloned_parents_data)
            },
            ProviderType::Children(ref _provider) => self
                .children
                .iter()
                .map(|child| child.calculate_expected_value() * child.get_probability())
                .sum::<f64>(),
        };
        self.current_expected_value
    }

    /// Returns the sum of regrets of all children.
    fn get_sum_regrets(&self) -> f64 {
        self.children.iter().map(|child| child.node.lock().unwrap().regret).sum()
    }

    /// Returns the total probability including parent probabilities.
    fn get_total_probability(&self) -> f64 {
        if self.parents_data.is_empty() {
            return self.probability; // If no parents, return the node's own probability
        }
        let parent_probability =
            self.parents_data.iter().fold(1.0, |acc, data| acc * data.get_probability());
        parent_probability * self.probability
    }

    /// Returns the children of this node.
    #[must_use]
    pub fn get_children(&self) -> Vec<WrappedRegret<UserData>> {
        self.children.clone()
    }

    /// Returns the user data associated with this node.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedUserData<UserData>> {
        self.provider.get_user_data()
    }

    /// Populates the children of this node using the provider.
    fn populate_children(&mut self) {
        if !self.children.is_empty() {
            return; // No children to populate
        }
        match self.provider.get_provider_type() {
            ProviderType::Children(ref provider) => {
                let mut cloned_parents_data = self.parents_data.clone();
                if let Some(ref user_data) = self.provider.get_user_data() {
                    cloned_parents_data.push(user_data.clone());
                } else {
                    // If no user data is provided, we just use the parents data
                }
                self.children = provider.get_children(cloned_parents_data);
                // populate children of children
                self.children.iter_mut().for_each(|child| {
                    child.populate_children();
                });
            },
            ProviderType::ExpectedValue(_) => {},
        }
    }

    /// Returns the current expected value.
    const fn get_expected_value(&self) -> f64 {
        self.current_expected_value
    }

    /// Returns the current probability.
    #[must_use]
    pub const fn get_probability(&self) -> f64 {
        self.probability
    }

    /// Returns the average probability.
    #[must_use]
    pub const fn get_average_probability(&self) -> f64 {
        self.average_probability
    }

    /// Returns the average expected value.
    #[must_use]
    pub const fn get_average_expected_value(&self) -> f64 {
        self.average_expected_value
    }
}

/// Thread-safe wrapper for a `RegretNode`.
#[derive(Clone)]
pub struct WrappedRegret<UserData: UserDataTrait> {
    /// The underlying regret node, wrapped in Arc<Mutex>.
    node: Arc<Mutex<RegretNode<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedRegret<UserData> {
    #[must_use]
    /// Creates a new wrapped regret node.
    pub fn new(node: RegretNode<UserData>) -> Self {
        Self { node: Arc::new(Mutex::new(node)) }
    }

    /// Returns the total probability including parent probabilities.
    #[must_use]
    pub fn get_total_probability(&self) -> f64 {
        safe_lock(&self.node).get_total_probability()
    }

    /// Returns the expected value of the node.
    #[must_use]
    pub fn get_expected_value(&self) -> f64 {
        safe_lock(&self.node).get_expected_value()
    }

    /// Calculates and returns the expected value of the node.
    #[must_use]
    pub fn calculate_expected_value(&self) -> f64 {
        safe_lock(&self.node).calculate_expected_value()
    }

    /// Returns the user data associated with the node.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedUserData<UserData>> {
        safe_lock(&self.node).get_user_data()
    }

    /// Returns the children of the node.
    #[must_use]
    pub fn get_children(&self) -> Vec<Self> {
        safe_lock(&self.node).get_children()
    }

    /// Calculates regrets for the node.
    pub fn calculate_regrets(
        &self,
        outer_expected_value: f64,
    ) {
        safe_lock(&self.node).calculate_regrets(outer_expected_value);
    }

    /// Calculates probabilities for the node.
    pub fn calculate_probabilities(
        &self,
        sum_regrets: f64,
        total_siblings: usize,
    ) {
        safe_lock(&self.node).calculate_probabilities(sum_regrets, total_siblings);
    }

    /// Returns the probability of the node.
    #[must_use]
    pub fn get_probability(&self) -> f64 {
        safe_lock(&self.node).get_probability()
    }

    /// Normalizes the probability of the node.
    pub fn calculate_normalized_probabilities(
        &self,
        total_probability: f64,
    ) {
        safe_lock(&self.node).calculate_normalized_probabilities(total_probability);
    }

    /// Returns the average probability of the node.
    #[must_use]
    pub fn get_average_probability(&self) -> f64 {
        safe_lock(&self.node).get_average_probability()
    }

    /// Returns the average expected value of the node.
    #[must_use]
    pub fn get_average_expected_value(&self) -> f64 {
        safe_lock(&self.node).get_average_expected_value()
    }

    /// Populates the children of the node.
    pub fn populate_children(&mut self) {
        safe_lock(&self.node).populate_children();
    }

    /// Updates the average values of the node.
    pub fn update_average_values(&mut self) {
        safe_lock(&self.node).update_average_values();
    }

    /// Returns a string representation of the node and its children.
    #[must_use]
    pub fn get_data_as_string(
        &self,
        indentation: usize,
    ) -> String {
        safe_lock(&self.node).get_data_as_string(indentation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enum representing choices in Rock-Paper-Scissors.
    #[derive(Default, Debug, Clone)]
    enum Choice {
        #[default]
        Rock,
        Paper,
        Scissors,
    }

    /// User data for Rock-Paper-Scissors.
    #[derive(Debug, Default, Clone)]
    struct RoshamboData {
        /// The chosen move.
        choice: Choice,
        /// Probability of this choice.
        probability: f64,
    }

    impl UserDataTrait for RoshamboData {
        /// Returns the probability of this choice.
        fn get_probability(&self) -> f64 {
            self.probability
        }
        /// Sets the probability of this choice.
        fn set_probability(
            &mut self,
            probability: f64,
        ) {
            self.probability = probability;
        }
        /// Returns a string representation of the choice.
        fn get_data_as_string(&self) -> String {
            format!("Choice: {:?}", self.choice.clone())
        }
    }

    /// Children provider for Rock-Paper-Scissors.
    #[derive(Debug, Clone)]
    struct RoshamboChildrenProvider {}

    impl RoshamboChildrenProvider {
        /// Creates a new children provider for Rock-Paper-Scissors.
        pub const fn new() -> Self {
            Self {}
        }
    }

    impl ChildrenProvider<RoshamboData> for RoshamboChildrenProvider {
        /// Returns the children nodes for Rock-Paper-Scissors.
        fn get_children(
            &self,
            parents_data: Vec<WrappedUserData<RoshamboData>>,
        ) -> Vec<WrappedRegret<RoshamboData>> {
            let probabilities = [0.4, 0.4, 0.2];
            match parents_data.len().cmp(&1) {
                std::cmp::Ordering::Less => {
                    let mut children = Vec::new();
                    for (i, choice) in
                        [Choice::Rock, Choice::Paper, Choice::Scissors].iter().enumerate()
                    {
                        let data = WrappedUserData::new(RoshamboData {
                            choice: choice.clone(),
                            probability: probabilities[i],
                        });
                        let node = RegretNode::new(
                            probabilities[i],
                            0.01,
                            parents_data.clone(),
                            WrappedProvider::new(Provider::new(
                                ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                                    Self::new(),
                                ))),
                                Some(data.clone()),
                            )),
                            None,
                        );
                        children.push(WrappedRegret::new(node));
                    }
                    children
                },
                std::cmp::Ordering::Equal => {
                    let mut children = Vec::new();
                    for (i, choice) in
                        [Choice::Rock, Choice::Paper, Choice::Scissors].iter().enumerate()
                    {
                        let data = WrappedUserData::new(RoshamboData {
                            choice: choice.clone(),
                            probability: probabilities[i],
                        });
                        let provider = Provider::new(
                            ProviderType::ExpectedValue(WrappedExpectedValueProvider::new(
                                Box::new(RoshamboExpectedValueProvider::new()),
                            )),
                            Some(data.clone()),
                        );
                        let node = RegretNode::new(
                            probabilities[i],
                            0.01,
                            parents_data.clone(),
                            WrappedProvider::new(provider),
                            None,
                        );
                        children.push(WrappedRegret::new(node));
                    }
                    children
                },
                std::cmp::Ordering::Greater => Vec::new(),
            }
        }
    }

    /// Expected value provider for Rock-Paper-Scissors.
    #[derive(Debug, Clone)]
    struct RoshamboExpectedValueProvider {}

    impl RoshamboExpectedValueProvider {
        /// Creates a new expected value provider for Rock-Paper-Scissors.
        pub const fn new() -> Self {
            Self {}
        }
    }

    impl ExpectedValueProvider<RoshamboData> for RoshamboExpectedValueProvider {
        /// Returns the expected value for Rock-Paper-Scissors.
        fn get_expected_value(
            &self,
            parents_data: Vec<WrappedUserData<RoshamboData>>,
        ) -> f64 {
            assert!(
                parents_data.len() >= 2,
                "Expected at least two parents data for expected value calculation"
            );
            let player_1_choice = &parents_data[parents_data.len() - 2].get_user_data().choice;
            let player_2_choice = &parents_data[parents_data.len() - 1].get_user_data().choice;
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

    /// Test for children provider in Rock-Paper-Scissors.
    #[test]
    fn test_roshambo_children_provider() {
        let provider = RoshamboChildrenProvider::new();
        let parents_data = vec![];
        let children = provider.get_children(parents_data);
        assert_eq!(children.len(), 3); // Should have three children for Rock, Paper, Scissors
    }

    /// Test for expected value provider in Rock-Paper-Scissors.
    #[test]
    fn test_roshambo_expected_value_provider() {
        let provider = RoshamboExpectedValueProvider::new();
        let parents_data = vec![
            WrappedUserData::new(RoshamboData { choice: Choice::Rock, probability: 0.333 }),
            WrappedUserData::new(RoshamboData { choice: Choice::Paper, probability: 0.333 }),
        ];
        let expected_value = provider.get_expected_value(parents_data);
        assert!((expected_value + 1.0).abs() < f64::EPSILON, "Paper beats Rock");
    }

    /// Test for regret minimization in Rock-Paper-Scissors.
    #[test]
    fn test_roshambo_regret_minimization() {
        let mut node = RegretNode::new(
            1.0,
            0.01,
            vec![],
            WrappedProvider::new(Provider::new(
                ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                    RoshamboChildrenProvider {},
                ))),
                None,
            )),
            Some(1.0),
        );

        node.solve(1000);
        // print node as string
        println!("{}", node.get_data_as_string(0));
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
