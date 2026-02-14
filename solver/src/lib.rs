//! # Solver Library
//!
//! This crate combines neural networks and regret minimization for solving optimization problems.
//! It provides a neural network-based solver that uses regret minimization algorithms.

#![warn(clippy::all)]
#![warn(clippy::style)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_crate_versions)]
#![warn(missing_docs)]

pub mod neural_children_provider;
pub mod neural_expected_value_provider;
pub mod neural_solver;
