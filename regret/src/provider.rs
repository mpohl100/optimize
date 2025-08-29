//! # Provider Module
//!
//! This module defines traits and wrappers for children and expected value providers
//! used in regret minimization trees. It also provides types for associating providers
//! with user data and for thread-safe access.

use crate::regret_node::WrappedRegret;
use crate::user_data::{UserDataTrait, WrappedUserData};
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

/// Trait for types that can generate child nodes given parent data.
pub trait ChildrenProvider<UserData: UserDataTrait, ChildType = WrappedRegret<UserData>>: std::fmt::Debug 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        parents_data: Vec<WrappedUserData<UserData>>,
    ) -> Vec<ChildType>;
}

/// Thread-safe wrapper for a boxed `ChildrenProvider`.
#[derive(Debug, Clone)]
pub struct WrappedChildrenProvider<UserData: UserDataTrait, ChildType = WrappedRegret<UserData>> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// The underlying provider, boxed and wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Box<dyn ChildrenProvider<UserData, ChildType>>>>,
}

impl<UserData: UserDataTrait, ChildType> WrappedChildrenProvider<UserData, ChildType> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// Creates a new wrapped children provider.
    #[must_use]
    pub fn new(provider: Box<dyn ChildrenProvider<UserData, ChildType>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the children nodes for the given parent data.
    #[must_use]
    pub fn get_children(
        &self,
        parents_data: Vec<WrappedUserData<UserData>>,
    ) -> Vec<ChildType> {
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

/// Enum representing either a children provider, expected value provider, or none (terminal node).
#[derive(Debug, Clone)]
pub enum ProviderType<UserData: UserDataTrait, ChildType = WrappedRegret<UserData>> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// Children provider variant.
    Children(WrappedChildrenProvider<UserData, ChildType>),
    /// Expected value provider variant.
    ExpectedValue(WrappedExpectedValueProvider<UserData>),
    /// None variant for terminal nodes.
    None,
}

/// Associates a provider type with optional user data.
#[derive(Debug, Clone)]
pub struct Provider<UserData: UserDataTrait, ChildType = WrappedRegret<UserData>> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// The provider type (children or expected value).
    pub provider_type: ProviderType<UserData, ChildType>,
    /// Optional user data associated with the provider.
    pub user_data: Option<WrappedUserData<UserData>>,
}

impl<UserData: UserDataTrait, ChildType> Provider<UserData, ChildType> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// Creates a new provider with the given type and optional user data.
    #[must_use]
    pub const fn new(
        provider_type: ProviderType<UserData, ChildType>,
        user_data: Option<WrappedUserData<UserData>>,
    ) -> Self {
        Self { provider_type, user_data }
    }
}

/// Thread-safe wrapper for a `Provider`.
#[derive(Debug, Clone)]
pub struct WrappedProvider<UserData: UserDataTrait, ChildType = WrappedRegret<UserData>> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// The underlying provider, wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Provider<UserData, ChildType>>>,
}

impl<UserData: UserDataTrait, ChildType> WrappedProvider<UserData, ChildType> 
where 
    ChildType: Clone + std::fmt::Debug,
{
    /// Creates a new wrapped provider.
    #[must_use]
    pub fn new(provider: Provider<UserData, ChildType>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the provider type.
    #[must_use]
    pub fn get_provider_type(&self) -> ProviderType<UserData, ChildType> {
        safe_lock(&self.provider).provider_type.clone()
    }

    /// Gets the optional user data.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedUserData<UserData>> {
        safe_lock(&self.provider).user_data.clone()
    }
}
