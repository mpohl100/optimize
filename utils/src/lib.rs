//! # Utility Functions Library
//!
//! Common utility functions and helpers used across the `ml_rust` project.
//! Provides safe abstractions and convenience functions for common operations.
//!
//! ## Overview
//!
//! This crate contains shared utilities including:
//! - **Safe concurrency primitives**: Poison-resistant mutex operations
//! - **Error handling helpers**: Common error patterns and utilities
//! - **Type conversions**: Safe conversions between numeric types
//! - **Validation functions**: Input validation and bounds checking
//!
//! ## Modules
//!
//! - [`safer`]: Safe abstractions for potentially unsafe operations
//!
//! ## Examples
//!
//! ### Safe Mutex Locking
//!
//! ```rust
//! use std::sync::Mutex;
//! use utils::safer::safe_lock;
//!
//! let mutex = Mutex::new(42);
//!
//! // Safe lock that handles poison errors gracefully
//! let guard = safe_lock(&mutex);
//! println!("Value: {}", *guard);
//! ```
//!
//! ## Design Philosophy
//!
//! This library follows these principles:
//! - **Fail gracefully**: Prefer degraded functionality over panics
//! - **Type safety**: Use Rust's type system to prevent errors at compile time
//! - **Performance**: Minimal overhead for common operations
//! - **Consistency**: Uniform API patterns across all utilities

#![warn(clippy::all)]
#![warn(clippy::style)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_crate_versions)]

pub mod safer;
