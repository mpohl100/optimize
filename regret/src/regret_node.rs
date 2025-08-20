use crate::provider::{ProviderType, WrappedProvider};
use crate::user_data::{UserDataTrait, WrappedUserData};
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

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
    // ...existing code for RegretNode<UserData>...
}

/// Thread-safe wrapper for a `RegretNode`.
#[derive(Clone)]
pub struct WrappedRegret<UserData: UserDataTrait> {
    /// The underlying regret node, wrapped in Arc<Mutex>.
    node: Arc<Mutex<RegretNode<UserData>>>,
}

impl<UserData: UserDataTrait> WrappedRegret<UserData> {
    // ...existing code for WrappedRegret<UserData>...
}
