# API Reference

This document provides a comprehensive reference for the ml_rust API, including all public modules, structs, functions, and traits.

## Overview

The ml_rust library is organized into several crates, each serving a specific purpose in the machine learning pipeline:

- **[alloc](#alloc-crate)**: Memory allocation management
- **[evol](#evol-crate)**: Evolutionary/genetic algorithms  
- **[matrix](#matrix-crate)**: Matrix operations and linear algebra
- **[neural](#neural-crate)**: Neural network implementation
- **[regret](#regret-crate)**: Regret minimization algorithms
- **[utils](#utils-crate)**: Common utility functions

## alloc Crate

Memory allocation management for performance-critical operations.

### AllocManager

Manages memory allocation with pooling and limits.

```rust
pub struct AllocManager {
    // Private fields
}

impl AllocManager {
    pub fn new(memory_limit: usize) -> Self
    pub fn allocate<T: Allocatable>(&mut self, item: &T) -> bool
    pub fn deallocate<T: Allocatable>(&mut self, item: &T)
    pub fn get_allocated_size(&self) -> usize
    pub fn get_max_allocated_size(&self) -> usize
}
```

**Example Usage:**
```rust
use alloc::AllocManager;

let mut manager = AllocManager::new(1024 * 1024); // 1MB limit
let success = manager.allocate(&some_matrix);
if success {
    // Use matrix...
    manager.deallocate(&some_matrix);
}
```

### Allocatable Trait

Trait for objects that can be managed by the allocation system.

```rust
pub trait Allocatable {
    fn allocate(&mut self);
    fn deallocate(&mut self);
    fn is_allocated(&self) -> bool;
    fn get_size(&self) -> usize;
    fn mark_for_use(&mut self);
    fn free_from_use(&mut self);
    fn is_in_use(&self) -> bool;
}
```

## evol Crate

Evolutionary algorithms for automated optimization and architecture search.

### Phenotype Trait

Core trait for objects that can evolve.

```rust
pub trait Phenotype: Clone + Send + Sync {
    type Config;
    
    fn random(config: &Self::Config) -> Self;
    fn mutate(&mut self, mutation_rate: f64);
    fn crossover(&self, other: &Self) -> Self;
    fn fitness(&self) -> f64;
    fn calculate_fitness(&mut self, data: &EvaluationData) -> Result<()>;
}
```

**Example Implementation:**
```rust
use evol::Phenotype;

#[derive(Clone)]
struct NetworkPhenotype {
    layers: Vec<LayerConfig>,
    fitness: Option<f64>,
}

impl Phenotype for NetworkPhenotype {
    type Config = NetworkConfig;
    
    fn random(config: &Self::Config) -> Self {
        // Generate random network architecture
        Self {
            layers: generate_random_layers(config),
            fitness: None,
        }
    }
    
    fn mutate(&mut self, mutation_rate: f64) {
        // Mutate network architecture
        for layer in &mut self.layers {
            if rand::random::<f64>() < mutation_rate {
                layer.mutate();
            }
        }
        self.fitness = None; // Reset fitness
    }
    
    fn crossover(&self, other: &Self) -> Self {
        // Combine two network architectures
        Self {
            layers: combine_layers(&self.layers, &other.layers),
            fitness: None,
        }
    }
    
    fn fitness(&self) -> f64 {
        self.fitness.unwrap_or(0.0)
    }
    
    fn calculate_fitness(&mut self, data: &EvaluationData) -> Result<()> {
        // Train and evaluate network
        let network = build_network(&self.layers)?;
        let accuracy = train_and_evaluate(network, data)?;
        self.fitness = Some(accuracy);
        Ok(())
    }
}
```

### EvolutionOptions

Configuration for evolutionary algorithms.

```rust
pub struct EvolutionOptions {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_count: usize,
    pub tournament_size: usize,
}

impl Default for EvolutionOptions {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 5,
            tournament_size: 3,
        }
    }
}
```

### RandomNumberGenerator

Thread-safe random number generation.

```rust
pub struct RandomNumberGenerator {
    // Private fields
}

impl RandomNumberGenerator {
    pub fn new() -> Self
    pub fn fetch_uniform(&mut self, min: f64, max: f64) -> f64
    pub fn fetch_normal(&mut self, mean: f64, std_dev: f64) -> f64
    pub fn fetch_bool(&mut self, probability: f64) -> bool
}
```

## matrix Crate

High-performance matrix operations for machine learning.

### WrappedMatrix

Thread-safe matrix implementation.

```rust
pub struct WrappedMatrix<T> {
    // Private fields - uses Arc<Mutex<Matrix<T>>>
}

impl<T> WrappedMatrix<T> 
where 
    T: Clone + Copy + Default + Send + Sync
{
    pub fn new(rows: usize, cols: usize) -> Self
    pub fn get_rows(&self) -> usize
    pub fn get_cols(&self) -> usize
    pub fn get_val(&self, row: usize, col: usize) -> Result<T, MatrixError>
    pub fn set_val(&mut self, row: usize, col: usize, val: T) -> Result<(), MatrixError>
    pub fn set_val_unchecked(&mut self, row: usize, col: usize, val: T)
    pub fn clone_data(&self) -> Matrix<T>
}
```

**Example Usage:**
```rust
use matrix::mat::WrappedMatrix;

// Create a 3x3 matrix
let mut matrix = WrappedMatrix::new(3, 3);

// Set values
matrix.set_val(0, 0, 1.0)?;
matrix.set_val(0, 1, 2.0)?;
matrix.set_val(1, 0, 3.0)?;

// Get values
let value = matrix.get_val(0, 0)?;
assert_eq!(value, 1.0);

// Get dimensions
assert_eq!(matrix.get_rows(), 3);
assert_eq!(matrix.get_cols(), 3);
```

### SumMatrix

Specialized matrix that maintains row sums efficiently.

```rust
pub struct SumMatrix<T> {
    // Private fields
}

impl<T> SumMatrix<T> 
where 
    T: Clone + Copy + Default + Send + Sync + Add<Output = T> + Sub<Output = T> + Div<Output = T>
{
    pub fn new(matrix: WrappedMatrix<T>) -> Self
    pub fn get_val(&self, row: usize, col: usize) -> Result<T, MatrixError>
    pub fn set_val(&mut self, row: usize, col: usize, val: T) -> Result<(), MatrixError>
    pub fn set_val_unchecked(&mut self, row: usize, col: usize, val: T)
    pub fn get_row_sum(&self, row: usize) -> T
    pub fn get_ratio(&self, row: usize, col: usize) -> Result<T, MatrixError>
}
```

**Example Usage:**
```rust
use matrix::{mat::WrappedMatrix, sum_mat::SumMatrix};

// Create base matrix
let base = WrappedMatrix::new(2, 3);
let mut sum_matrix = SumMatrix::new(base);

// Set values - row sums are maintained automatically
sum_matrix.set_val(0, 0, 1.0)?;
sum_matrix.set_val(0, 1, 2.0)?;
sum_matrix.set_val(0, 2, 3.0)?;

// Get row sum
let sum = sum_matrix.get_row_sum(0); // 6.0

// Get normalized ratios (useful for probabilities)
let ratio = sum_matrix.get_ratio(0, 0)?; // 1.0/6.0 ≈ 0.167
```

### MatrixError

Error types for matrix operations.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    IndexOutOfBounds { row: usize, col: usize, max_rows: usize, max_cols: usize },
    DimensionMismatch { expected: (usize, usize), actual: (usize, usize) },
    InvalidOperation(String),
    AllocationFailure,
}

impl std::fmt::Display for MatrixError { /* ... */ }
impl std::error::Error for MatrixError { /* ... */ }
```

## neural Crate

Complete neural network implementation with training and inference.

### Layer Types

#### DenseLayer

Fully connected neural network layer.

```rust
pub struct DenseLayer {
    // Private fields
}

impl DenseLayer {
    pub fn new(
        input_size: usize, 
        output_size: usize, 
        activation: ActivationFunction
    ) -> Result<Self, NeuralNetworkError>
    
    pub fn forward(&self, input: &Matrix<f64>) -> Result<Matrix<f64>, NeuralNetworkError>
    pub fn backward(&mut self, input: &Matrix<f64>, grad_output: &Matrix<f64>) -> Result<Matrix<f64>, NeuralNetworkError>
    pub fn get_weights(&self) -> &Matrix<f64>
    pub fn get_bias(&self) -> &Vector<f64>
    pub fn update_weights(&mut self, learning_rate: f64)
}
```

### ActivationFunction

Enumeration of supported activation functions.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

impl ActivationFunction {
    pub fn apply(&self, x: f64) -> f64
    pub fn derivative(&self, x: f64) -> f64
    pub fn apply_vector(&self, input: &[f64]) -> Vec<f64>
}
```

**Example Usage:**
```rust
use neural::activation::ActivationFunction;

let relu = ActivationFunction::ReLU;
assert_eq!(relu.apply(2.0), 2.0);
assert_eq!(relu.apply(-1.0), 0.0);

let sigmoid = ActivationFunction::Sigmoid;
let result = sigmoid.apply(0.0); // ≈ 0.5
```

### NeuralNetwork

Complete neural network implementation.

```rust
pub struct NeuralNetwork {
    // Private fields
}

impl NeuralNetwork {
    pub fn new(layers: Vec<LayerConfig>) -> Result<Self, NeuralNetworkError>
    pub fn forward(&self, input: &Matrix<f64>) -> Result<Matrix<f64>, NeuralNetworkError>
    pub fn train(&mut self, data: &TrainingData, epochs: usize) -> Result<(), NeuralNetworkError>
    pub fn predict(&self, input: &Matrix<f64>) -> Result<Matrix<f64>, NeuralNetworkError>
    pub fn save(&self, path: &Path) -> Result<(), NeuralNetworkError>
    pub fn load(path: &Path) -> Result<Self, NeuralNetworkError>
    pub fn evaluate(&self, test_data: &TestData) -> Result<f64, NeuralNetworkError>
}
```

### TrainingSession

Manages the training process with configuration and monitoring.

```rust
pub struct TrainingSession {
    // Private fields
}

impl TrainingSession {
    pub fn new(
        model_dir: &Path,
        data_file: &Path,
        target_file: &Path,
        params: TrainingParams,
    ) -> Result<Self, NeuralNetworkError>
    
    pub fn train(&mut self) -> Result<(), NeuralNetworkError>
    pub fn get_training_progress(&self) -> TrainingProgress
    pub fn save_checkpoint(&self) -> Result<(), NeuralNetworkError>
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(), NeuralNetworkError>
}
```

### TrainingParams

Configuration for training sessions.

```rust
pub struct TrainingParams {
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub validation_split: f64,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub optimizer: OptimizerConfig,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            epochs: 1000,
            learning_rate: 0.001,
            batch_size: 32,
            validation_split: 0.2,
            early_stopping: Some(EarlyStoppingConfig::default()),
            optimizer: OptimizerConfig::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        }
    }
}
```

## regret Crate

Regret minimization algorithms for decision-making under uncertainty.

### RegretMinimizer

Core regret minimization implementation.

```rust
pub struct RegretMinimizer<T> {
    // Private fields
}

impl<T> RegretMinimizer<T> 
where 
    T: Children + ExpectedValue + Send + Sync
{
    pub fn new(root: T) -> Self
    pub fn run_iterations(&mut self, iterations: usize) -> Result<(), RegretError>
    pub fn get_strategy(&self) -> Strategy
    pub fn get_best_response(&self) -> &T
    pub fn get_average_strategy(&self) -> Strategy
}
```

### Children Trait

Trait for objects that have child nodes in game trees.

```rust
pub trait Children {
    type Child;
    
    fn children(&self) -> Vec<Self::Child>;
    fn is_terminal(&self) -> bool;
}
```

### ExpectedValue Trait

Trait for calculating expected values in game scenarios.

```rust
pub trait ExpectedValue {
    fn expected_value(&self, strategy: &Strategy) -> f64;
    fn utility(&self, outcome: &Outcome) -> f64;
}
```

## utils Crate

Common utility functions used across the project.

### safer Module

Safe abstractions for potentially unsafe operations.

```rust
pub fn safe_lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T>
```

Safely locks a mutex, handling poison errors gracefully by extracting the inner value.

**Example Usage:**
```rust
use std::sync::Mutex;
use utils::safer::safe_lock;

let data = Mutex::new(42);
let guard = safe_lock(&data);
println!("Value: {}", *guard);
```

## Error Handling

All crates follow consistent error handling patterns:

### Common Error Types

- **MatrixError**: Matrix operation errors
- **NeuralNetworkError**: Neural network operation errors  
- **AllocationError**: Memory allocation errors
- **RegretError**: Regret minimization errors

### Error Handling Pattern

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Operation failed: {reason}")]
    OperationFailed { reason: String },
    
    #[error("IO error")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MyError>;
```

## Threading and Concurrency

### Thread Safety

- All matrix types use `Arc<Mutex<T>>` for thread safety
- Genetic algorithms support parallel fitness evaluation
- Training can use thread pools for batch processing

### Usage with Threads

```rust
use std::sync::Arc;
use std::thread;

let matrix = Arc::new(WrappedMatrix::new(100, 100));

let handles: Vec<_> = (0..4)
    .map(|i| {
        let matrix = Arc::clone(&matrix);
        thread::spawn(move || {
            // Each thread can safely access the matrix
            let value = matrix.get_val(i, i).unwrap();
            println!("Thread {}: value = {}", i, value);
        })
    })
    .collect();

for handle in handles {
    handle.join().unwrap();
}
```

## Performance Considerations

### Memory Management

- Use the `alloc` crate for memory-intensive operations
- Prefer `Vec::with_capacity()` when size is known
- Consider memory pooling for frequent allocations

### Computational Efficiency

- Matrix operations use SIMD when available
- Training supports batch processing with configurable batch sizes
- Genetic algorithms can evaluate fitness in parallel

### Best Practices

1. **Batch Operations**: Process data in batches for better cache locality
2. **Memory Reuse**: Reuse allocated matrices when possible
3. **Parallel Processing**: Use `rayon` for CPU-intensive operations
4. **Profiling**: Use `cargo flamegraph` and `perf` for performance analysis

## Integration Examples

### Complete Neural Network Training

```rust
use neural::*;
use std::path::Path;

fn train_xor_network() -> Result<(), Box<dyn std::error::Error>> {
    // Create training configuration
    let params = TrainingParams {
        epochs: 2000,
        learning_rate: 0.1,
        batch_size: 4,
        ..Default::default()
    };
    
    // Set up training session
    let mut session = TrainingSession::new(
        Path::new("models/xor"),
        Path::new("data/xor_inputs.csv"),
        Path::new("data/xor_targets.csv"),
        params,
    )?;
    
    // Train the network
    session.train()?;
    
    // Evaluate performance
    let network = session.get_network();
    let accuracy = network.evaluate(&test_data)?;
    println!("Final accuracy: {:.2}%", accuracy * 100.0);
    
    Ok(())
}
```

### Genetic Algorithm for Architecture Search

```rust
use evol::*;
use neural::*;

fn evolve_network_architecture() -> Result<(), Box<dyn std::error::Error>> {
    // Define evolution parameters
    let options = EvolutionOptions {
        population_size: 50,
        generations: 20,
        mutation_rate: 0.15,
        crossover_rate: 0.8,
        elite_count: 5,
        tournament_size: 3,
    };
    
    // Create initial population
    let config = NetworkConfig::new(784, 10); // MNIST-like problem
    let mut population: Vec<NetworkPhenotype> = (0..options.population_size)
        .map(|_| NetworkPhenotype::random(&config))
        .collect();
    
    // Evolution loop
    for generation in 0..options.generations {
        // Evaluate fitness for all individuals
        for individual in &mut population {
            individual.calculate_fitness(&training_data)?;
        }
        
        // Sort by fitness
        population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        
        // Create next generation
        let mut next_generation = Vec::new();
        
        // Keep elite individuals
        next_generation.extend(population[..options.elite_count].iter().cloned());
        
        // Generate offspring through crossover and mutation
        while next_generation.len() < options.population_size {
            let parent1 = tournament_select(&population, options.tournament_size);
            let parent2 = tournament_select(&population, options.tournament_size);
            
            let mut offspring = parent1.crossover(parent2);
            offspring.mutate(options.mutation_rate);
            
            next_generation.push(offspring);
        }
        
        population = next_generation;
        
        println!("Generation {}: Best fitness = {:.4}", 
                generation, population[0].fitness());
    }
    
    // Return best individual
    let best_network = &population[0];
    println!("Evolution complete! Best fitness: {:.4}", best_network.fitness());
    
    Ok(())
}
```

This API reference provides a comprehensive overview of the ml_rust library's public interface. For more detailed examples and usage patterns, see the [TUTORIALS.md](TUTORIALS.md) document.