//! # User Data Module
//!
//! This module defines the trait for user-defined data used in regret nodes,
//! and provides a thread-safe wrapper for such data. Implement `UserDataTrait`
//! for your domain-specific data to use with the regret minimization framework.

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
    #[must_use]
    pub fn get_probability(&self) -> f64 {
        safe_lock(&self.user_data).get_probability()
    }

    /// Gets the string representation of the inner user data.
    #[must_use]
    pub fn get_data_as_string(&self) -> String {
        safe_lock(&self.user_data).get_data_as_string()
    }
}
