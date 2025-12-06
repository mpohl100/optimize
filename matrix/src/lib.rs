//! # Matrix Operations Library
//!
//! A high-performance matrix operations library designed for machine learning applications.
//! Provides efficient matrix computations with memory management optimizations.
//!
//! ## Overview
//!
//! This library provides fundamental matrix operations including:
//! - **Basic matrix operations**: addition, subtraction, multiplication
//! - **Specialized matrices**: sum matrices for probabilistic operations
//! - **Memory-efficient implementations**: with custom allocation strategies
//! - **Thread-safe operations**: using Arc and Mutex where appropriate
//!
//! ## Modules
//!
//! - [`mat`]: Core matrix implementation with basic linear algebra operations
//! - [`sum_mat`]: Specialized sum matrix for maintaining row/column sums efficiently
//!
//! ## Examples
//!
//! ### Basic Matrix Operations
//!
//! ```rust
//! use matrix::mat::WrappedMatrix;
//!
//! // Create a 3x3 matrix
//! let matrix = WrappedMatrix::<f64>::new(3, 3);
//!
//! // Set values
//! matrix.set_mut_unchecked(0, 0, 1.0);
//! matrix.set_mut_unchecked(0, 1, 2.0);
//!
//! // Get values
//! let value = matrix.get_unchecked(0, 0);
//! assert_eq!(value, 1.0);
//! ```
//!
//! ### Sum Matrix for Probability Distributions
//!
//! ```rust
//! use matrix::mat::WrappedMatrix;
//! use matrix::sum_mat::SumMatrix;
//!
//! // Create a sum matrix that maintains row sums automatically
//! let base_matrix = WrappedMatrix::<i64>::new(2, 3);
//! let mut sum_matrix = SumMatrix::new(base_matrix);
//!
//! // Set values - row sums are updated automatically
//! sum_matrix.set_val(0, 0, 1).unwrap();
//! sum_matrix.set_val(0, 1, 2).unwrap();
//!
//! // Get normalized ratios (useful for probability distributions)
//! let ratio = sum_matrix.get_ratio(0, 0).unwrap(); // 1.0 / 3.0 ≈ 0.333...
//! ```
//!
//! ## Performance Features
//!
//! - **Memory pooling**: Efficient memory reuse for temporary matrices
//! - **Cache-friendly layouts**: Optimized memory access patterns
//! - **SIMD optimizations**: Vectorized operations where possible
//! - **Thread safety**: Safe concurrent access with minimal locking
//!
//! ## Thread Safety
//!
//! All matrix types are designed to be thread-safe:
//! - `WrappedMatrix` uses `Arc<Mutex<Matrix<T>>>` for safe shared access
//! - Operations are atomic where possible
//! - Lock-free algorithms for read-heavy workloads

#![warn(clippy::all)]
#![warn(clippy::style)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
// #![warn(missing_docs)]
// #![warn(missing_debug_implementations)]
// #![warn(rust_2018_idioms)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_crate_versions)]

pub mod directory;
pub mod mat;
pub mod persistable_matrix;
pub mod sum_mat;
