use crate::regret_node::WrappedRegret;
use crate::user_data::{UserDataTrait, WrappedUserData};
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

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
