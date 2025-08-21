//! # Neural Network Library
//!
//! A comprehensive neural network library for Rust, designed for experimentation with
//! automated neural network architecture search using genetic algorithms.
//!
//! ## Overview
//!
//! This library provides a complete neural network implementation including:
//! - **Dense (fully connected) layers** with various activation functions
//! - **Training algorithms** with gradient descent and optimization
//! - **Genetic algorithm integration** for automated architecture search
//! - **Memory management** with custom allocation strategies
//! - **Model persistence** with YAML-based serialization
//!
//! ## Architecture
//!
//! The library is organized into several modules:
//!
//! - [`activation`]: Activation functions (ReLU, Sigmoid, Tanh, etc.)
//! - [`layer`]: Neural network layer implementations
//! - [`nn`]: Complete neural network structures and builders
//! - [`training`]: Training algorithms and data management
//! - [`utilities`]: Memory management and utility functions
//!
//! ## Performance Considerations
//!
//! The library is designed for performance with:
//! - Custom memory allocators for managing large matrices
//! - Parallel processing using thread pools
//! - SIMD-optimized operations where possible
//! - Memory-mapped file I/O for large datasets
//!
//! ## Feature Flags
//!
//! - `default`: Basic neural network functionality
//! - `serde`: Serialization support for model persistence
//! - `parallel`: Multi-threaded training and inference
//! - `gpu`: GPU acceleration (when available)

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

pub mod activation;
pub mod layer;
pub mod nn;
pub mod training;
pub mod utilities;
