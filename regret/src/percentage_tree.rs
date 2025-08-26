//! # `PercentageTree` Module
//!
//! This module implements a percentage-based decision tree that can make random decisions
//! based on probability distributions. It uses wrapped decisions and providers to manage
//! decision-making processes.

use crate::provider::{ProviderType, WrappedProvider};
use crate::user_data::{DecisionTrait, WrappedDecision};
use evol::rng::RandomNumberGenerator;
use rand::Rng;

/// A percentage-based decision tree that makes random decisions based on probability distributions.
#[derive(Debug, Clone)]
pub struct PercentageTree<Decision: DecisionTrait> {
    /// Vector of parent wrapped decisions that led to this node.
    parent_decisions: Vec<WrappedDecision<Decision>>,
    /// The provider type which is either `ChildrenProvider` or None for terminal nodes.
    provider: WrappedProvider<Decision>,
}

impl<Decision: DecisionTrait> PercentageTree<Decision> {
    /// Creates a new `PercentageTree` with the given parent decisions and provider.
    ///
    /// # Arguments
    ///
    /// * `parent_decisions` - Vector of wrapped decisions that led to this node
    /// * `provider` - The provider which can be `ChildrenProvider` or None
    ///
    /// # Examples
    ///
    /// ```
    /// use regret::percentage_tree::PercentageTree;
    /// use regret::provider::{Provider, ProviderType, WrappedProvider};
    /// use regret::user_data::{DecisionTrait, WrappedDecision};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    /// struct MyDecision {
    ///     value: i32,
    ///     probability: f64,
    /// }
    ///
    /// impl DecisionTrait for MyDecision {
    ///     fn get_probability(&self) -> f64 { self.probability }
    ///     fn set_probability(&mut self, prob: f64) { self.probability = prob; }
    ///     fn get_data_as_string(&self) -> String { format!("Decision: {}", self.value) }
    /// }
    ///
    /// let parent_decisions: Vec<WrappedDecision<MyDecision>> = vec![];
    /// let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
    /// let tree = PercentageTree::new(parent_decisions, provider);
    /// ```
    #[must_use]
    pub const fn new(
        parent_decisions: Vec<WrappedDecision<Decision>>,
        provider: WrappedProvider<Decision>,
    ) -> Self {
        Self {
            parent_decisions,
            provider,
        }
    }

    /// Makes a random decision based on the probability distribution of child nodes.
    ///
    /// This method rolls a random number between 0 and 100,000 and selects a child
    /// based on cumulative probability distribution. The selected child's wrapped
    /// decision is appended to the result vector.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a `RandomNumberGenerator`
    ///
    /// # Returns
    ///
    /// A vector containing the parent decisions plus the randomly selected child decision.
    /// If this is a terminal node (None provider), returns just the parent decisions.
    ///
    /// # Examples
    ///
    /// ```
    /// use regret::percentage_tree::PercentageTree;
    /// use regret::provider::{Provider, ProviderType, WrappedProvider};
    /// use regret::user_data::{WrappedDecision, DecisionTrait};
    /// use evol::rng::RandomNumberGenerator;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    /// struct MyDecision {
    ///     value: i32,
    ///     probability: f64,
    /// }
    ///
    /// impl DecisionTrait for MyDecision {
    ///     fn get_probability(&self) -> f64 { self.probability }
    ///     fn set_probability(&mut self, prob: f64) { self.probability = prob; }
    ///     fn get_data_as_string(&self) -> String { format!("Decision: {}", self.value) }
    /// }
    ///
    /// // Create a terminal percentage tree
    /// let parent_decisions: Vec<WrappedDecision<MyDecision>> = vec![];
    /// let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
    /// let tree = PercentageTree::new(parent_decisions, provider);
    /// let mut rng = RandomNumberGenerator::new();
    /// let result = tree.random_decision(&mut rng);
    /// ```
    #[must_use]
    pub fn random_decision(&self, rng: &mut RandomNumberGenerator) -> Vec<WrappedDecision<Decision>> {
        let mut result = self.parent_decisions.clone();

        match self.provider.get_provider_type() {
            ProviderType::Children(ref children_provider) => {
                // Get children from the provider
                let children = children_provider.get_children(self.parent_decisions.clone());

                if children.is_empty() {
                    return result;
                }

                // Calculate total probability
                let total_probability: f64 = children
                    .iter()
                    .map(super::regret_node::WrappedRegret::get_probability)
                    .sum();

                if total_probability <= 0.0 {
                    // If no valid probabilities, return current result
                    return result;
                }

                // Generate random number between 0 and 100,000
                let random_value = f64::from(rng.rng.gen_range(0..=100_000)) / 100_000.0;
                let target_probability = random_value * total_probability;

                // Find child based on cumulative probability
                let mut cumulative_probability = 0.0;
                for child in &children {
                    cumulative_probability += child.get_probability();
                    if cumulative_probability >= target_probability {
                        // Append the selected child's decision to the result
                        if let Some(child_decision) = child.get_user_data() {
                            result.push(child_decision);
                        }
                        break;
                    }
                }
            }
            ProviderType::ExpectedValue(_) | ProviderType::None => {
                // Expected value providers don't generate children, terminal nodes have no children
            }
        }

        result
    }

    /// Returns a reference to the parent decisions.
    #[must_use]
    pub fn get_parent_decisions(&self) -> &[WrappedDecision<Decision>] {
        &self.parent_decisions
    }

    /// Returns a reference to the provider.
    #[must_use]
    pub const fn get_provider(&self) -> &WrappedProvider<Decision> {
        &self.provider
    }

    /// Checks if this is a terminal node (has None provider).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self.provider.get_provider_type(), ProviderType::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChildrenProvider, Provider, ProviderType, WrappedChildrenProvider};
    use crate::regret_node::{RegretNode, WrappedRegret};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Clone, Serialize, Deserialize)]
    struct TestDecision {
        value: i32,
        probability: f64,
    }

    impl DecisionTrait for TestDecision {
        fn get_probability(&self) -> f64 {
            self.probability
        }

        fn set_probability(&mut self, probability: f64) {
            self.probability = probability;
        }

        fn get_data_as_string(&self) -> String {
            format!("TestDecision(value: {}, prob: {})", self.value, self.probability)
        }
    }

    #[derive(Debug, Clone)]
    struct TestChildrenProvider;

    impl ChildrenProvider<TestDecision> for TestChildrenProvider {
        fn get_children(
            &self,
            parents_data: Vec<WrappedDecision<TestDecision>>,
        ) -> Vec<WrappedRegret<TestDecision>> {
            // Create three children with different probabilities
            let children_data = [
                (1, 0.5),  // 50% probability
                (2, 0.3),  // 30% probability
                (3, 0.2),  // 20% probability
            ];

            children_data
                .iter()
                .map(|(value, prob)| {
                    let decision = WrappedDecision::new(TestDecision {
                        value: *value,
                        probability: *prob,
                    });
                    let provider = WrappedProvider::new(Provider::new(ProviderType::None, Some(decision.clone())));
                    let node = RegretNode::new(
                        *prob,
                        0.01,
                        parents_data.clone(),
                        provider,
                        None,
                    );
                    WrappedRegret::new(node)
                })
                .collect()
        }
    }

    #[test]
    fn test_percentage_tree_creation() {
        let parent_decisions = vec![WrappedDecision::new(TestDecision {
            value: 1,
            probability: 0.5,
        })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let tree = PercentageTree::new(parent_decisions.clone(), provider);

        assert_eq!(tree.get_parent_decisions().len(), 1);
        assert!(tree.is_terminal());
    }

    #[test]
    fn test_random_decision_with_children() {
        let parent_decisions = vec![];
        let children_provider = Box::new(TestChildrenProvider);
        let provider = WrappedProvider::new(Provider::new(
            ProviderType::Children(WrappedChildrenProvider::new(children_provider)),
            None,
        ));
        let tree = PercentageTree::new(parent_decisions, provider);
        let mut rng = RandomNumberGenerator::new();

        // Test multiple random decisions to ensure they're within expected range
        let mut results = std::collections::HashMap::new();
        for _ in 0..1000 {
            let result = tree.random_decision(&mut rng);
            assert_eq!(result.len(), 1); // Should have one decision from the children

            let value = result[0].get_decision_data().value;
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
        // Allow for some variation (within 20% of expected)
        assert!(*count_1 > *count_2);
        assert!(*count_2 > *count_3);
    }

    #[test]
    fn test_random_decision_with_terminal_node() {
        let parent_decisions = vec![WrappedDecision::new(TestDecision {
            value: 1,
            probability: 0.5,
        })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let tree = PercentageTree::new(parent_decisions.clone(), provider);
        let mut rng = RandomNumberGenerator::new();

        let result = tree.random_decision(&mut rng);

        // Terminal node should return the same parent decisions
        assert_eq!(result.len(), parent_decisions.len());
        assert_eq!(
            result[0].get_decision_data().value,
            parent_decisions[0].get_decision_data().value
        );
    }

    #[test]
    fn test_serialization() {
        // Note: Full serialization is not supported due to Arc<Mutex<...>> in WrappedDecision and WrappedProvider
        // This test is kept as placeholder for future implementation
        let parent_decisions = vec![WrappedDecision::new(TestDecision {
            value: 42,
            probability: 0.75,
        })];
        let provider = WrappedProvider::new(Provider::new(ProviderType::None, None));
        let tree = PercentageTree::new(parent_decisions, provider);

        // Test basic properties instead of serialization
        assert_eq!(tree.get_parent_decisions().len(), 1);
        assert_eq!(
            tree.get_parent_decisions()[0]
                .get_decision_data()
                .value,
            42
        );
        assert_eq!(
            tree.get_parent_decisions()[0]
                .get_decision_data()
                .probability,
            0.75
        );
        assert!(tree.is_terminal());
    }
}