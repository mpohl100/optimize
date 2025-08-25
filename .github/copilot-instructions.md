# GitHub Copilot Instructions for ml_rust

This document provides comprehensive guidance for GitHub Copilot when working on the ml_rust machine learning library. These instructions ensure code consistency, quality, and adherence to modern Rust best practices.

## Repository Overview

ml_rust is an experimental machine learning library written in Rust that focuses on neural network automation using genetic algorithms. The repository is organized as a Cargo workspace with multiple specialized crates.

### Workspace Structure

```
ml_rust/
├── alloc/          # Memory allocation management
├── app/            # CLI applications (train, evaluate, predict, nn_generator)
├── evol/           # Evolutionary/genetic algorithms
├── gen/            # Neural network generation utilities
├── markov/         # Markov chain implementation
├── matrix/         # Matrix operations and linear algebra
├── neural/         # Core neural network implementation
├── regret/         # Regret minimization algorithms
└── utils/          # Shared utility functions
```

## Code Style and Standards

### Rust Edition and Features

- **Use Rust Edition 2021** for all new code
- **Enable comprehensive linting** with these attributes in lib.rs:
  ```rust
  #![warn(clippy::all)]
  #![warn(clippy::style)]
  #![warn(clippy::pedantic)]
  #![warn(clippy::nursery)]
  #![warn(clippy::cargo)]
  #![warn(missing_docs)]
  #![warn(missing_debug_implementations)]
  #![warn(rust_2018_idioms)]
  #![forbid(unsafe_code)] // Unless specifically needed for performance
  ```

### Naming Conventions

- **Types**: Use `PascalCase` (e.g., `NeuralNetwork`, `TrainingParams`)
- **Functions/Variables**: Use `snake_case` (e.g., `calculate_gradient`, `learning_rate`)
- **Constants**: Use `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_LEARNING_RATE`)
- **Modules**: Use `snake_case` (e.g., `neural_network`, `training_session`)

### Error Handling

**Always use the `thiserror` crate for custom errors:**

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NeuralNetworkError {
    #[error("Invalid layer dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },
    
    #[error("Training data mismatch: {0}")]
    DataMismatch(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_yaml::Error),
}

pub type Result<T> = std::result::Result<T, NeuralNetworkError>;
```

**Error Handling Patterns:**
- Use `Result<T, E>` for all fallible operations
- Use `?` operator for error propagation
- Provide meaningful error messages with context
- Use `anyhow` for application-level error handling in binaries
- Use `thiserror` for library-level custom errors

### Performance and Memory Management

**Memory Allocation:**
- Use the custom `alloc` crate for memory-intensive operations
- Prefer `Vec::with_capacity()` when size is known
- Use `Box<[T]>` for fixed-size arrays
- Consider `smallvec` for small collections

**Performance Patterns:**
```rust
// Prefer iterators over loops
let result: Vec<_> = data
    .iter()
    .filter(|&x| x > threshold)
    .map(|x| x * 2.0)
    .collect();

// Use `const` for compile-time constants
const DEFAULT_LEARNING_RATE: f64 = 0.001;

// Use #[inline] for small, frequently called functions
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

### Logging and Tracing

**Use the `tracing` ecosystem for structured logging:**

```rust
use tracing::{info, warn, error, debug, trace, instrument};

#[instrument(skip(data))]
pub fn train_network(
    network: &mut NeuralNetwork,
    data: &TrainingData,
    epochs: usize,
) -> Result<()> {
    info!("Starting training with {} epochs", epochs);
    
    for epoch in 0..epochs {
        debug!("Training epoch {}", epoch);
        // Training logic...
        
        if epoch % 100 == 0 {
            info!("Completed epoch {}/{}", epoch, epochs);
        }
    }
    
    info!("Training completed successfully");
    Ok(())
}
```

## Documentation Standards

### Doc Comments

**Every public item must have comprehensive documentation:**

```rust
/// A dense (fully connected) neural network layer.
///
/// This layer performs a linear transformation followed by an activation function:
/// `output = activation(input * weights + bias)`
///
/// # Examples
///
/// ```
/// use neural::layer::DenseLayer;
/// use neural::activation::ActivationFunction;
///
/// let layer = DenseLayer::new(784, 128, ActivationFunction::ReLU)?;
/// let input = Matrix::zeros(1, 784);
/// let output = layer.forward(&input)?;
/// assert_eq!(output.cols(), 128);
/// ```
///
/// # Performance
///
/// The forward pass has O(input_size * output_size) complexity.
/// Memory usage is O(input_size * output_size) for weights storage.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weight matrix with shape (input_size, output_size)
    weights: Matrix<f64>,
    /// Bias vector with length output_size
    bias: Vector<f64>,
    /// Activation function to apply after linear transformation
    activation: ActivationFunction,
}
```

### Module Documentation

**Each module should have comprehensive module-level docs:**

```rust
//! Neural network layer implementations.
//!
//! This module provides various types of neural network layers including:
//! - Dense (fully connected) layers
//! - Activation layers
//! - Normalization layers
//!
//! # Architecture
//!
//! All layers implement the [`Layer`] trait which provides:
//! - Forward propagation via [`Layer::forward`]
//! - Backward propagation via [`Layer::backward`]
//! - Parameter access via [`Layer::parameters`]
//!
//! # Examples
//!
//! ```
//! use neural::layer::{DenseLayer, ActivationFunction};
//! use neural::matrix::Matrix;
//!
//! // Create a simple network
//! let layer1 = DenseLayer::new(784, 128, ActivationFunction::ReLU)?;
//! let layer2 = DenseLayer::new(128, 10, ActivationFunction::Softmax)?;
//! ```
```

## Testing Strategy

### Unit Tests

**Write comprehensive unit tests for all functions:**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(2, 3, ActivationFunction::Linear).unwrap();
        let input = Matrix::from_vec(vec![vec![1.0, 2.0]]);
        
        let output = layer.forward(&input).unwrap();
        
        assert_eq!(output.rows(), 1);
        assert_eq!(output.cols(), 3);
    }

    #[test]
    fn test_invalid_dimensions() {
        let result = DenseLayer::new(0, 10, ActivationFunction::ReLU);
        assert!(result.is_err());
    }

    // Property-based testing
    proptest! {
        #[test]
        fn test_layer_output_dimensions(
            input_size in 1..100usize,
            output_size in 1..100usize,
            batch_size in 1..32usize,
        ) {
            let layer = DenseLayer::new(input_size, output_size, ActivationFunction::Linear)?;
            let input = Matrix::zeros(batch_size, input_size);
            
            let output = layer.forward(&input)?;
            
            prop_assert_eq!(output.rows(), batch_size);
            prop_assert_eq!(output.cols(), output_size);
        }
    }
}
```

### Integration Tests

**Create integration tests in `tests/` directories:**

```rust
// tests/neural_network_integration.rs
use neural::*;
use matrix::Matrix;

#[test]
fn test_xor_learning() {
    let mut network = NeuralNetworkBuilder::new()
        .add_dense_layer(2, 4, ActivationFunction::ReLU)
        .add_dense_layer(4, 1, ActivationFunction::Sigmoid)
        .build()
        .unwrap();

    let inputs = Matrix::from_vec(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);
    
    let targets = Matrix::from_vec(vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ]);

    network.train(&inputs, &targets, 1000).unwrap();

    // Test predictions
    let predictions = network.predict(&inputs).unwrap();
    for i in 0..4 {
        let predicted = predictions.get(i, 0).unwrap();
        let target = targets.get(i, 0).unwrap();
        assert!((predicted - target).abs() < 0.1, 
                "XOR not learned correctly: {} vs {}", predicted, target);
    }
}
```

### Benchmark Tests

**Add benchmarks for performance-critical code:**

```rust
// benches/matrix_operations.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use matrix::Matrix;

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    for size in [64, 128, 256, 512].iter() {
        let a = Matrix::random(*size, *size);
        let b = Matrix::random(*size, *size);
        
        group.bench_with_input(
            format!("{}x{}", size, size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a) * black_box(&b);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_matrix_multiplication);
criterion_main!(benches);
```

## Modern Rust Libraries and Dependencies

### Recommended Dependencies

**Core Libraries:**
```toml
[dependencies]
# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1.0"

# Logging and tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Numerical computing
nalgebra = "0.33"  # Linear algebra
ndarray = "0.16"   # N-dimensional arrays
num-traits = "0.2" # Numeric traits

# Random number generation
rand = "0.9"
rand_distr = "0.5"

# CLI applications
clap = { version = "4.0", features = ["derive"] }

# Async (if needed)
tokio = { version = "1.0", features = ["full"] }

# Parallel processing
rayon = "1.11"

# File system operations
walkdir = "2.0"
tempfile = "3.0"

[dev-dependencies]
# Testing
proptest = "1.0"
approx = "0.5"
criterion = { version = "0.5", features = ["html_reports"] }

# Test utilities
assert_matches = "1.5"
```

**Specialized ML Libraries:**
```toml
# Machine Learning specific
candle = "0.6"      # PyTorch-like ML framework
tch = "0.17"        # PyTorch bindings
smartcore = "0.3"   # ML algorithms
linfa = "0.7"       # Comprehensive ML toolkit
```

### Feature Flags

**Use feature flags for optional functionality:**

```toml
[features]
default = ["std"]
std = []
serde = ["dep:serde", "dep:serde_yaml"]
parallel = ["dep:rayon"]
gpu = ["dep:candle"]
python-bindings = ["dep:pyo3"]

[dependencies]
pyo3 = { version = "0.22", optional = true }
candle = { version = "0.6", optional = true }
```

## Architecture Patterns

### Builder Pattern

**Use the builder pattern for complex configurations:**

```rust
#[derive(Debug)]
pub struct NeuralNetworkBuilder {
    layers: Vec<LayerConfig>,
    optimizer: OptimizerConfig,
    loss_function: LossFunction,
}

impl NeuralNetworkBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: OptimizerConfig::Adam { learning_rate: 0.001 },
            loss_function: LossFunction::MeanSquaredError,
        }
    }

    pub fn add_dense_layer(
        mut self,
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
    ) -> Self {
        self.layers.push(LayerConfig::Dense {
            input_size,
            output_size,
            activation,
        });
        self
    }

    pub fn optimizer(mut self, optimizer: OptimizerConfig) -> Self {
        self.optimizer = optimizer;
        self
    }

    pub fn build(self) -> Result<NeuralNetwork> {
        // Validation and construction logic
        Ok(NeuralNetwork::from_config(self)?)
    }
}
```

### Type-Safe Configuration

**Use enums and structs for type-safe configuration:**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerConfig {
    SGD { learning_rate: f64 },
    Adam { learning_rate: f64, beta1: f64, beta2: f64 },
    RMSprop { learning_rate: f64, decay: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub optimizer: OptimizerConfig,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub validation_split: f64,
}
```

### Trait-Based Design

**Use traits for extensibility and testability:**

```rust
/// Trait for neural network layers
pub trait Layer: Send + Sync + std::fmt::Debug {
    /// Forward propagation
    fn forward(&self, input: &Matrix<f64>) -> Result<Matrix<f64>>;
    
    /// Backward propagation
    fn backward(
        &mut self,
        input: &Matrix<f64>,
        grad_output: &Matrix<f64>,
    ) -> Result<Matrix<f64>>;
    
    /// Get trainable parameters
    fn parameters(&self) -> Vec<&Matrix<f64>>;
    
    /// Get mutable trainable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Matrix<f64>>;
    
    /// Get output shape given input shape
    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>>;
}

/// Trait for optimizers
pub trait Optimizer: Send + Sync + std::fmt::Debug {
    /// Update parameters based on gradients
    fn step(&mut self, parameters: &mut [Matrix<f64>], gradients: &[Matrix<f64>]) -> Result<()>;
    
    /// Reset optimizer state
    fn reset(&mut self);
}
```

## Memory Management and Performance

### Custom Allocators

**Use the custom allocation manager for memory-intensive operations:**

```rust
use crate::alloc::AllocManager;

#[derive(Debug)]
pub struct TrainingSession {
    alloc_manager: AllocManager,
    // other fields...
}

impl TrainingSession {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            alloc_manager: AllocManager::new(memory_limit),
        }
    }

    pub fn train_batch(&mut self, batch: &TrainingBatch) -> Result<()> {
        // Allocate temporary matrices
        let temp_matrix = self.alloc_manager.allocate_matrix(batch.size(), self.input_dim)?;
        
        // Use matrix for computations
        // ...
        
        // Matrix is automatically deallocated when dropped
        Ok(())
    }
}
```

### SIMD and Vectorization

**Use SIMD when possible for numerical computations:**

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized dot product for f32 vectors
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_feature = "avx2")]
    unsafe {
        // AVX2 implementation
        dot_product_avx2(a, b)
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        // Fallback to standard implementation
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
```

## Genetic Algorithm Patterns

### Phenotype Design

**Use the established phenotype pattern for evolutionary algorithms:**

```rust
use evol::{Phenotype, EvolutionOptions};

#[derive(Debug, Clone)]
pub struct NetworkPhenotype {
    network: NeuralNetwork,
    fitness: Option<f64>,
}

impl Phenotype for NetworkPhenotype {
    type Config = NetworkConfig;

    fn random(config: &Self::Config) -> Self {
        Self {
            network: NeuralNetwork::random(config),
            fitness: None,
        }
    }

    fn mutate(&mut self, mutation_rate: f64) {
        self.network.mutate(mutation_rate);
        self.fitness = None; // Reset fitness after mutation
    }

    fn crossover(&self, other: &Self) -> Self {
        Self {
            network: self.network.crossover(&other.network),
            fitness: None,
        }
    }

    fn fitness(&self) -> f64 {
        self.fitness.unwrap_or(0.0)
    }

    fn calculate_fitness(&mut self, evaluation_data: &EvaluationData) -> Result<()> {
        self.fitness = Some(self.network.evaluate(evaluation_data)?);
        Ok(())
    }
}
```

## CLI Application Patterns

### Structured Commands

**Use clap derive for CLI applications:**

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Global verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train a neural network
    Train {
        /// Path to the model directory
        #[arg(short, long)]
        model_dir: std::path::PathBuf,
        
        /// Path to training data CSV
        #[arg(short, long)]
        data: std::path::PathBuf,
        
        /// Path to target data CSV
        #[arg(short, long)]
        targets: std::path::PathBuf,
        
        /// Number of training epochs
        #[arg(short, long, default_value_t = 1000)]
        epochs: usize,
    },
    /// Evaluate a trained network
    Evaluate {
        /// Path to the model directory
        #[arg(short, long)]
        model_dir: std::path::PathBuf,
        
        /// Path to test data CSV
        #[arg(short, long)]
        data: std::path::PathBuf,
    },
    /// Generate network architectures
    Generate {
        /// Number of networks to generate
        #[arg(short, long, default_value_t = 10)]
        count: usize,
        
        /// Output directory
        #[arg(short, long)]
        output: std::path::PathBuf,
    },
}
```

## Continuous Integration Requirements

### Code Quality Gates

**All code must pass these quality gates:**

1. **Formatting**: `cargo fmt --check`
2. **Linting**: `cargo clippy -- -D warnings`
3. **Testing**: `cargo test --all-features`
4. **Documentation**: `cargo doc --no-deps --all-features`
5. **Benchmarks**: `cargo bench` (for performance regressions)

### Coverage Requirements

**Maintain high test coverage:**
- Minimum 80% line coverage for new code
- All public APIs must have tests
- Integration tests for all CLI commands
- Property-based tests for mathematical functions

## Security Considerations

### Input Validation

**Always validate inputs rigorously:**

```rust
pub fn load_training_data(path: &Path) -> Result<TrainingData> {
    // Validate file exists and is readable
    if !path.exists() {
        return Err(NeuralNetworkError::FileNotFound(path.to_path_buf()));
    }
    
    if !path.is_file() {
        return Err(NeuralNetworkError::InvalidFileType(path.to_path_buf()));
    }
    
    // Validate file size (prevent DoS attacks)
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > MAX_FILE_SIZE {
        return Err(NeuralNetworkError::FileTooLarge(metadata.len()));
    }
    
    // Load and validate data format
    let data = std::fs::read_to_string(path)?;
    let parsed = parse_csv(&data)?;
    
    // Validate data dimensions and ranges
    validate_training_data(&parsed)?;
    
    Ok(parsed)
}
```

### Safe Defaults

**Use safe defaults for all configurations:**

```rust
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            batch_size: 32,
            learning_rate: 0.001,
            // Safe defaults that won't cause issues
            max_gradient_norm: Some(1.0),
            weight_decay: 0.0001,
            early_stopping: Some(EarlyStoppingConfig::default()),
        }
    }
}
```

## Documentation Generation

### API Documentation

**Generate comprehensive API docs:**

```bash
# Generate docs with examples
cargo doc --no-deps --all-features --document-private-items

# Serve docs locally for review
cargo doc --no-deps --all-features --open
```

### Architecture Documentation

**Maintain architecture docs in `docs/` directory:**
- `docs/ARCHITECTURE.md`: High-level system design
- `docs/API.md`: API reference and examples
- `docs/DEVELOPMENT.md`: Development setup and guidelines
- `docs/TUTORIALS.md`: Usage tutorials and examples

## Final Notes

- **Performance First**: This is a machine learning library; performance is critical
- **Type Safety**: Use Rust's type system to prevent runtime errors
- **Documentation**: Every public item needs comprehensive documentation
- **Testing**: Write tests first, implement second
- **Benchmarks**: Monitor performance regressions with benchmarks
- **Error Handling**: Provide clear, actionable error messages
- **Memory Management**: Use the custom allocator for large data structures
- **Parallelization**: Use rayon for CPU-bound parallel operations

When implementing new features, always consider:
1. How will this be tested?
2. How will this be documented?
3. What errors can occur and how should they be handled?
4. What are the performance implications?
5. How does this fit into the overall architecture?

Remember: Code is read more often than it's written. Make it clear, well-documented, and maintainable.

## Agent Mode Development Workflow

**When working in agent mode, always follow this iterative workflow for every change:**

1. Run `cargo build` to check for compilation errors.
2. Run `cargo fmt` to ensure code formatting.
3. Run `cargo clippy --fix` to automatically fix lint warnings.
4. Run `cargo clippy -- -D warnings` to ensure no warnings remain.
5. Fix any remaining Clippy warnings manually.
6. Repeat steps 1–5 until all warnings are resolved and the code builds cleanly.

**No code change is complete until all Clippy warnings are fixed and the code is formatted.**