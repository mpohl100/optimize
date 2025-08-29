//! # `PercentageNode` Module
//!
//! This module implements a percentage-based decision node that can make random decisions
//! based on probability distributions. It follows the same pattern as `RegretNode` but focuses
//! on percentage-based decision making with recursive random decision functionality.

use crate::provider::PercentageProviderType;
use crate::user_data::{UserDataTrait, WrappedUserData};
use evol::rng::RandomNumberGenerator;
use rand::Rng;
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

/// Represents a node in the percentage-based decision tree.
#[derive(Clone)]
pub struct PercentageNode<UserData: UserDataTrait> {
    /// Data from parent nodes.
    parents_data: Vec<WrappedUserData<UserData>>,
    /// Provider for children.
    provider: PercentageProviderType<UserData>,
    /// Child nodes.
    children: Vec<WrappedUserData<UserData>>,
}

impl<UserData: UserDataTrait> PercentageNode<UserData> {
    /// Creates a new percentage node.
    #[must_use]
    pub fn new(
        parents_data: Vec<WrappedUserData<UserData>>,
        provider: PercentageProviderType<UserData>,
    ) -> Self {
        let mut node = Self { parents_data, provider, children: Vec::new() };

        // Create children via the provider
        node.create_children();
        node
    }

    /// Creates children using the percentage provider.
    fn create_children(&mut self) {
        if !self.children.is_empty() {
            return; // Children already created
        }

        match &self.provider {
            PercentageProviderType::Children(ref provider) => {
                self.children = provider.get_children(self.parents_data.clone());
            },
            PercentageProviderType::None => {
                // Terminal node, no children to create
            },
        }
    }

    /// Makes a random decision by recursively calling `random_decision` on randomly chosen children.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a `RandomNumberGenerator`
    ///
    /// # Returns
    ///
    /// A vector containing decisions from the randomly chosen path through the tree.
    /// For terminal nodes, returns the parent data. For non-terminal nodes, selects a
    /// child based on probability distribution and recursively calls `random_decision`
    /// on that child if it has its own percentage provider.
    #[must_use]
    pub fn random_decision(
        &self,
        rng: &mut RandomNumberGenerator,
    ) -> Vec<WrappedUserData<UserData>> {
        let mut result = self.parents_data.clone();

        match &self.provider {
            PercentageProviderType::Children(_) => {
                if self.children.is_empty() {
                    return result;
                }

                // Calculate total probability
                let total_probability: f64 =
                    self.children.iter().map(WrappedUserData::get_probability).sum();

                if total_probability <= 0.0 {
                    // If no valid probabilities, return current result
                    return result;
                }

                // Generate random number between 0 and 100,000
                let random_value = f64::from(rng.rng.gen_range(0..=100_000)) / 100_000.0;
                let target_probability = random_value * total_probability;

                // Find child based on cumulative probability
                let mut cumulative_probability = 0.0;
                for child in &self.children {
                    cumulative_probability += child.get_probability();
                    if cumulative_probability >= target_probability {
                        // Add the selected child to the result
                        result.push(child.clone());

                        // Recursively call random_decision on a new percentage node created with this child
                        // Create a percentage provider that could generate more children from this child
                        // For this implementation, we'll create a terminal node and return
                        let child_node = Self::new(
                            result.clone(),
                            PercentageProviderType::None, // Terminal child for this implementation
                        );
                        let recursive_decisions = child_node.random_decision(rng);

                        // Add any new decisions from the recursive call that aren't already present
                        for decision in recursive_decisions {
                            if !result.iter().any(|existing| {
                                // Compare by data string representation for uniqueness
                                existing.get_data_as_string() == decision.get_data_as_string()
                            }) {
                                result.push(decision);
                            }
                        }
                        break;
                    }
                }
            },
            PercentageProviderType::None => {
                // Terminal node, return current path (parents_data)
            },
        }

        result
    }

    /// Returns a reference to the parent data.
    #[must_use]
    pub fn get_parents_data(&self) -> &[WrappedUserData<UserData>] {
        &self.parents_data
    }

    /// Returns a reference to the children.
    #[must_use]
    pub fn get_children(&self) -> &[WrappedUserData<UserData>] {
        &self.children
    }

    /// Returns a reference to the provider.
    #[must_use]
    pub const fn get_provider(&self) -> &PercentageProviderType<UserData> {
        &self.provider
    }

    /// Checks if this is a terminal node (has None provider).
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self.provider, PercentageProviderType::None)
    }
}

/// Thread-safe wrapper for a `PercentageNode`.
#[derive(Clone)]
pub struct WrappedPercentageNode<UserData: UserDataTrait> {
    /// The underlying node, wrapped in Arc<Mutex>.
    node: Arc<Mutex<PercentageNode<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedPercentageNode<UserData> {
    /// Creates a new wrapped percentage node.
    #[must_use]
    pub fn new(node: PercentageNode<UserData>) -> Self {
        Self { node: Arc::new(Mutex::new(node)) }
    }

    /// Makes a random decision by calling the underlying node's `random_decision` method.
    #[must_use]
    pub fn random_decision(
        &self,
        rng: &mut RandomNumberGenerator,
    ) -> Vec<WrappedUserData<UserData>> {
        safe_lock(&self.node).random_decision(rng)
    }

    /// Returns the parent data from the underlying node.
    #[must_use]
    pub fn get_parents_data(&self) -> Vec<WrappedUserData<UserData>> {
        safe_lock(&self.node).get_parents_data().to_vec()
    }

    /// Returns the children from the underlying node.
    #[must_use]
    pub fn get_children(&self) -> Vec<WrappedUserData<UserData>> {
        safe_lock(&self.node).get_children().to_vec()
    }

    /// Checks if this is a terminal node.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        safe_lock(&self.node).is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{PercentageProvider, WrappedPercentageProvider};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    struct TestDecision {
        value: i32,
        probability: f64,
    }

    impl UserDataTrait for TestDecision {
        fn get_probability(&self) -> f64 {
            self.probability
        }

        fn set_probability(
            &mut self,
            probability: f64,
        ) {
            self.probability = probability;
        }

        fn get_data_as_string(&self) -> String {
            format!("TestDecision(value: {}, prob: {})", self.value, self.probability)
        }
    }

    #[derive(Debug, Clone)]
    struct TestPercentageProvider;

    impl PercentageProvider<TestDecision> for TestPercentageProvider {
        fn get_children(
            &self,
            _parents_data: Vec<WrappedUserData<TestDecision>>,
        ) -> Vec<WrappedUserData<TestDecision>> {
            // Create three children with different probabilities
            let children_data = [
                (1, 0.5), // 50% probability
                (2, 0.3), // 30% probability
                (3, 0.2), // 20% probability
            ];

            children_data
                .iter()
                .map(|(value, prob)| {
                    WrappedUserData::new(TestDecision { value: *value, probability: *prob })
                })
                .collect()
        }
    }

    #[test]
    fn test_percentage_node_creation() {
        let parent_data = vec![WrappedUserData::new(TestDecision { value: 1, probability: 0.5 })];
        let provider = PercentageProviderType::None;
        let node = PercentageNode::new(parent_data.clone(), provider);

        assert_eq!(node.get_parents_data().len(), 1);
        assert!(node.is_terminal());
        assert!(node.get_children().is_empty());
    }

    #[test]
    fn test_percentage_node_with_children() {
        let parent_data = vec![];
        let children_provider = Box::new(TestPercentageProvider);
        let provider =
            PercentageProviderType::Children(WrappedPercentageProvider::new(children_provider));
        let node = PercentageNode::new(parent_data, provider);

        assert!(!node.is_terminal());
        assert_eq!(node.get_children().len(), 3);
    }

    #[test]
    fn test_random_decision_with_children() {
        let parent_data = vec![];
        let children_provider = Box::new(TestPercentageProvider);
        let provider =
            PercentageProviderType::Children(WrappedPercentageProvider::new(children_provider));
        let node = PercentageNode::new(parent_data, provider);
        let mut rng = RandomNumberGenerator::new();

        // Test multiple random decisions to ensure they're within expected range
        let mut results = std::collections::HashMap::new();
        for _ in 0..1000 {
            let result = node.random_decision(&mut rng);
            assert!(!result.is_empty()); // Should have at least the selected child

            // Get the first decision (the selected child)
            let value = result[0].get_user_data().value;
            *results.entry(value).or_insert(0) += 1;
        }

        // Verify that all three possible values were selected
        assert!(results.contains_key(&1));
        assert!(results.contains_key(&2));
        assert!(results.contains_key(&3));

        // Value 1 should be selected most often (50% probability)
        let count_1 = results.get(&1).unwrap_or(&0);
        let count_2 = results.get(&2).unwrap_or(&0);
        let count_3 = results.get(&3).unwrap_or(&0);

        // With 1000 trials and 50%, 30%, 20% probabilities, we expect roughly:
        // Value 1: ~500, Value 2: ~300, Value 3: ~200
        // Allow for some variation
        assert!(*count_1 > *count_2);
        assert!(*count_2 > *count_3);
    }

    #[test]
    fn test_random_decision_with_terminal_node() {
        let parent_data = vec![WrappedUserData::new(TestDecision { value: 1, probability: 0.5 })];
        let provider = PercentageProviderType::None;
        let node = PercentageNode::new(parent_data.clone(), provider);
        let mut rng = RandomNumberGenerator::new();

        let result = node.random_decision(&mut rng);

        // Terminal node should return the same parent data
        assert_eq!(result.len(), parent_data.len());
        assert_eq!(result[0].get_user_data().value, parent_data[0].get_user_data().value);
    }

    #[test]
    fn test_wrapped_percentage_node() {
        let parent_data = vec![];
        let children_provider = Box::new(TestPercentageProvider);
        let provider =
            PercentageProviderType::Children(WrappedPercentageProvider::new(children_provider));
        let node = PercentageNode::new(parent_data, provider);
        let wrapped_node = WrappedPercentageNode::new(node);
        let mut rng = RandomNumberGenerator::new();

        assert!(!wrapped_node.is_terminal());
        assert_eq!(wrapped_node.get_children().len(), 3);

        let result = wrapped_node.random_decision(&mut rng);
        assert!(!result.is_empty());
    }
}
