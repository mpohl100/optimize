//! Safe abstractions for potentially unsafe operations.
//!
//! This module provides safer alternatives to operations that could panic or behave
//! unexpectedly in concurrent environments.

use std::sync::{Mutex, MutexGuard};

/// Safely locks a mutex, handling poison errors gracefully.
///
/// This function provides a safe way to lock a mutex that handles poison errors
/// by extracting the inner value. This is useful in scenarios where you want to
/// continue execution even if another thread panicked while holding the lock.
///
/// # Arguments
///
/// * `m` - A reference to the mutex to lock
///
/// # Returns
///
/// A `MutexGuard` that provides access to the protected data.
///
/// # Examples
///
/// ```rust
/// use std::sync::Mutex;
/// use utils::safer::safe_lock;
///
/// let data = Mutex::new(vec![1, 2, 3]);
/// let guard = safe_lock(&data);
/// println!("Data length: {}", guard.len());
/// ```
///
/// # Safety
///
/// This function is safe to use even if the mutex is poisoned. It will extract
/// the inner value from a poison error, allowing continued access to the data.
/// However, be aware that the data might be in an inconsistent state if the
/// previous holder of the lock panicked during a critical section.
pub fn safe_lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}
