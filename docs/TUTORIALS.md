# Tutorials and Usage Examples

This document provides step-by-step tutorials for using the ml_rust machine learning library, covering everything from basic matrix operations to advanced neural network training and genetic algorithm optimization.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Matrix Operations](#basic-matrix-operations)
3. [Neural Network Basics](#neural-network-basics)
4. [Training Your First Model](#training-your-first-model)
5. [Genetic Algorithm Optimization](#genetic-algorithm-optimization)
6. [Advanced Features](#advanced-features)
7. [Performance Optimization](#performance-optimization)
8. [Real-World Examples](#real-world-examples)

## Getting Started

### Installation

Add ml_rust to your `Cargo.toml`:

```toml
[dependencies]
matrix = { git = "https://github.com/mpohl100/ml_rust", path = "matrix" }
neural = { git = "https://github.com/mpohl100/ml_rust", path = "neural" }
evol = { git = "https://github.com/mpohl100/ml_rust", path = "evol" }
utils = { git = "https://github.com/mpohl100/ml_rust", path = "utils" }
```

### Basic Project Setup

Create a new Rust project:

```bash
cargo new ml_tutorial
cd ml_tutorial
```

Update your `Cargo.toml`:

```toml
[package]
name = "ml_tutorial"
version = "0.1.0"
edition = "2021"

[dependencies]
matrix = { path = "../ml_rust/matrix" }
neural = { path = "../ml_rust/neural" }
evol = { path = "../ml_rust/evol" }
utils = { path = "../ml_rust/utils" }
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
```

## Basic Matrix Operations

### Creating and Manipulating Matrices

```rust
use matrix::mat::WrappedMatrix;
use matrix::sum_mat::SumMatrix;

fn basic_matrix_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3x3 matrix of f64 values
    let matrix = WrappedMatrix::<f64>::new(3, 3);
    
    // Set values (unchecked for performance)
    matrix.set_mut_unchecked(0, 0, 1.0);
    matrix.set_mut_unchecked(0, 1, 2.0);
    matrix.set_mut_unchecked(0, 2, 3.0);
    matrix.set_mut_unchecked(1, 0, 4.0);
    matrix.set_mut_unchecked(1, 1, 5.0);
    matrix.set_mut_unchecked(1, 2, 6.0);
    
    // Get values
    let value = matrix.get_unchecked(0, 0);
    println!("Matrix[0,0] = {}", value);
    
    // Get dimensions
    println!("Matrix dimensions: {}x{}", matrix.rows(), matrix.cols());
    
    // Create a sum matrix for probability calculations
    let base_matrix = WrappedMatrix::<i64>::new(2, 3);
    let mut sum_matrix = SumMatrix::new(base_matrix);
    
    // Set values - row sums are maintained automatically
    sum_matrix.set_val(0, 0, 10)?;
    sum_matrix.set_val(0, 1, 20)?;
    sum_matrix.set_val(0, 2, 30)?;
    
    // Get row sum
    let row_sum = sum_matrix.get_row_sum(0);
    println!("Row 0 sum: {}", row_sum); // 60
    
    // Get normalized ratios (probabilities)
    let prob_0 = sum_matrix.get_ratio(0, 0)?;
    let prob_1 = sum_matrix.get_ratio(0, 1)?;
    let prob_2 = sum_matrix.get_ratio(0, 2)?;
    
    println!("Probabilities: {:.3}, {:.3}, {:.3}", prob_0, prob_1, prob_2);
    // Output: 0.167, 0.333, 0.500
    
    Ok(())
}
```

### Thread-Safe Matrix Operations

```rust
use std::sync::Arc;
use std::thread;
use matrix::mat::WrappedMatrix;

fn parallel_matrix_tutorial() {
    let matrix = Arc::new(WrappedMatrix::<f64>::new(100, 100));
    let mut handles = vec![];
    
    // Initialize matrix in parallel
    for thread_id in 0..4 {
        let matrix = Arc::clone(&matrix);
        let handle = thread::spawn(move || {
            let start_row = thread_id * 25;
            let end_row = (thread_id + 1) * 25;
            
            for i in start_row..end_row {
                for j in 0..100 {
                    let value = (i * 100 + j) as f64;
                    matrix.set_mut_unchecked(i, j, value);
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify results
    let value = matrix.get_unchecked(50, 25);
    println!("Matrix[50,25] = {}", value); // 5025.0
}
```

## Neural Network Basics

### Understanding Network Architecture

Neural networks in ml_rust are defined using YAML configuration files that specify the layers and their properties.

Create a file `simple_network.yaml`:

```yaml
layers:
- layer_type: !Dense
    input_size: 4
    output_size: 8
  activation: ReLU
- layer_type: !Dense
    input_size: 8
    output_size: 4
  activation: ReLU
- layer_type: !Dense
    input_size: 4
    output_size: 1
  activation: Sigmoid
```

### Loading and Creating Networks

```rust
use neural::nn::shape::NeuralNetworkShape;
use neural::nn::neuralnet::{ClassicNeuralNetwork, TrainableClassicNeuralNetwork};
use neural::utilities::util::{Utils, WrappedUtils};
use std::path::Path;

fn create_network_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    // Load network shape from YAML
    let shape_yaml = std::fs::read_to_string("simple_network.yaml")?;
    let shape: NeuralNetworkShape = serde_yaml::from_str(&shape_yaml)?;
    
    println!("Network shape loaded:");
    println!("Input size: {}", shape.layers[0].layer_type.get_input_size());
    println!("Output size: {}", shape.layers.last().unwrap().layer_type.get_output_size());
    println!("Hidden layers: {}", shape.layers.len() - 1);
    
    // Create utilities for memory management
    let utils = Utils::new(1024 * 1024, 4); // 1MB memory limit, 4 threads
    let wrapped_utils = WrappedUtils::new(utils);
    
    // Create a trainable network
    let model_dir = Path::new("./model");
    std::fs::create_dir_all(model_dir)?;
    
    // Save the shape to the model directory
    let shape_path = model_dir.join("shape.yaml");
    std::fs::write(&shape_path, &shape_yaml)?;
    
    println!("Network configuration saved to: {:?}", shape_path);
    
    Ok(())
}
```

### Activation Functions

```rust
use neural::activation::ActivationFunction;

fn activation_function_tutorial() {
    let functions = vec![
        ActivationFunction::ReLU,
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
        ActivationFunction::Linear,
    ];
    
    let test_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    
    for func in &functions {
        println!("\n{:?} activation:", func);
        for &x in &test_values {
            let y = func.apply(x);
            let dy = func.derivative(x);
            println!("  f({:.1}) = {:.3}, f'({:.1}) = {:.3}", x, y, x, dy);
        }
    }
}
```

## Training Your First Model

### Preparing Data

First, let's create some sample training data for the XOR problem:

```rust
use std::fs::File;
use std::io::Write;

fn create_xor_dataset() -> Result<(), Box<dyn std::error::Error>> {
    // Create input data (XOR inputs)
    let mut input_file = File::create("xor_inputs.csv")?;
    writeln!(input_file, "0.0,0.0")?;
    writeln!(input_file, "0.0,1.0")?;
    writeln!(input_file, "1.0,0.0")?;
    writeln!(input_file, "1.0,1.0")?;
    
    // Create target data (XOR outputs)
    let mut target_file = File::create("xor_targets.csv")?;
    writeln!(target_file, "0.0")?;
    writeln!(target_file, "1.0")?;
    writeln!(target_file, "1.0")?;
    writeln!(target_file, "0.0")?;
    
    println!("XOR dataset created: xor_inputs.csv, xor_targets.csv");
    Ok(())
}
```

Create the network configuration for XOR (`xor_network.yaml`):

```yaml
layers:
- layer_type: !Dense
    input_size: 2
    output_size: 4
  activation: ReLU
- layer_type: !Dense
    input_size: 4
    output_size: 1
  activation: Sigmoid
```

### Training Session

```rust
use neural::training::training_session::TrainingSession;
use neural::training::training_params::TrainingParams;
use std::path::Path;

fn train_xor_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    // Create training parameters
    let params = TrainingParams {
        epochs: 2000,
        learning_rate: 0.1,
        print_every: 200,
        max_cpu_percentage: 80.0,
        ..Default::default()
    };
    
    // Set up paths
    let model_dir = Path::new("./xor_model");
    let shape_file = Path::new("xor_network.yaml");
    let input_file = Path::new("xor_inputs.csv");
    let target_file = Path::new("xor_targets.csv");
    
    // Create model directory
    std::fs::create_dir_all(model_dir)?;
    
    // Create training session
    let mut session = TrainingSession::new(
        model_dir,
        Some(shape_file),
        input_file,
        target_file,
        params,
    )?;
    
    println!("Starting XOR training...");
    
    // Train the network
    session.train()?;
    
    println!("Training completed! Model saved to: {:?}", model_dir);
    
    Ok(())
}
```

### Evaluating the Trained Model

```rust
use neural::nn::nn_factory::neural_network_from_disk;
use neural::training::data_importer::DataImporter;
use std::path::Path;

fn evaluate_xor_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new("./xor_model");
    
    // Load the trained network
    let network = neural_network_from_disk(model_dir)?;
    
    // Load test data
    let input_file = Path::new("xor_inputs.csv");
    let target_file = Path::new("xor_targets.csv");
    
    let inputs = DataImporter::import_data(input_file)?;
    let targets = DataImporter::import_data(target_file)?;
    
    println!("Evaluating XOR network:");
    println!("Input -> Expected | Predicted");
    println!("------------------------");
    
    for i in 0..inputs.get_rows() {
        let input_row = inputs.get_row(i)?;
        let target_value = targets.get_val(i, 0)?;
        
        // Make prediction
        let prediction = network.predict(&input_row)?;
        let predicted_value = prediction.get_val(0, 0)?;
        
        println!("{:.1},{:.1} -> {:.1} | {:.3}", 
                 input_row.get_val(0, 0)?, 
                 input_row.get_val(0, 1)?,
                 target_value,
                 predicted_value);
    }
    
    Ok(())
}
```

## Genetic Algorithm Optimization

### Creating a Custom Phenotype

```rust
use evol::{Phenotype, EvolutionOptions};
use neural::nn::shape::NeuralNetworkShape;
use neural::activation::ActivationFunction;
use neural::layer::dense_layer::LayerType;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NetworkConfig {
    input_size: usize,
    output_size: usize,
    max_hidden_layers: usize,
    max_layer_size: usize,
}

#[derive(Clone, Debug)]
struct NetworkPhenotype {
    shape: NeuralNetworkShape,
    fitness: Option<f64>,
}

impl Phenotype for NetworkPhenotype {
    type Config = NetworkConfig;
    
    fn random(config: &Self::Config) -> Self {
        use evol::rng::RandomNumberGenerator;
        let mut rng = RandomNumberGenerator::new();
        
        let mut layers = Vec::new();
        let num_hidden = rng.fetch_uniform(1.0, config.max_hidden_layers as f64) as usize;
        
        let mut prev_size = config.input_size;
        
        // Add hidden layers
        for _ in 0..num_hidden {
            let size = rng.fetch_uniform(2.0, config.max_layer_size as f64) as usize;
            let activation = match rng.fetch_uniform(0.0, 3.0) as usize {
                0 => ActivationFunction::ReLU,
                1 => ActivationFunction::Sigmoid,
                _ => ActivationFunction::Tanh,
            };
            
            layers.push(neural::nn::shape::LayerShape {
                layer_type: LayerType::Dense {
                    input_size: prev_size,
                    output_size: size,
                },
                activation,
            });
            
            prev_size = size;
        }
        
        // Add output layer
        layers.push(neural::nn::shape::LayerShape {
            layer_type: LayerType::Dense {
                input_size: prev_size,
                output_size: config.output_size,
            },
            activation: ActivationFunction::Sigmoid,
        });
        
        Self {
            shape: NeuralNetworkShape { layers },
            fitness: None,
        }
    }
    
    fn mutate(&mut self, mutation_rate: f64) {
        use evol::rng::RandomNumberGenerator;
        let mut rng = RandomNumberGenerator::new();
        
        // Mutate each layer with given probability
        for layer in &mut self.shape.layers[..self.shape.layers.len()-1] { // Don't mutate output layer
            if rng.fetch_uniform(0.0, 1.0) < mutation_rate {
                // Mutate layer size
                if let LayerType::Dense { input_size, output_size } = &mut layer.layer_type {
                    let new_size = rng.fetch_uniform(2.0, 64.0) as usize;
                    *output_size = new_size;
                }
                
                // Potentially mutate activation
                if rng.fetch_uniform(0.0, 1.0) < 0.3 {
                    layer.activation = match rng.fetch_uniform(0.0, 3.0) as usize {
                        0 => ActivationFunction::ReLU,
                        1 => ActivationFunction::Sigmoid,
                        _ => ActivationFunction::Tanh,
                    };
                }
            }
        }
        
        // Fix layer connections
        self.fix_connections();
        self.fitness = None; // Reset fitness after mutation
    }
    
    fn crossover(&self, other: &Self) -> Self {
        use evol::rng::RandomNumberGenerator;
        let mut rng = RandomNumberGenerator::new();
        
        // Simple crossover: take layers from both parents
        let min_layers = self.shape.layers.len().min(other.shape.layers.len());
        let mut new_layers = Vec::new();
        
        for i in 0..min_layers {
            let layer = if rng.fetch_uniform(0.0, 1.0) < 0.5 {
                self.shape.layers[i].clone()
            } else {
                other.shape.layers[i].clone()
            };
            new_layers.push(layer);
        }
        
        let mut offspring = Self {
            shape: NeuralNetworkShape { layers: new_layers },
            fitness: None,
        };
        
        offspring.fix_connections();
        offspring
    }
    
    fn fitness(&self) -> f64 {
        self.fitness.unwrap_or(0.0)
    }
    
    fn calculate_fitness(&mut self, _data: &()) -> Result<(), Box<dyn std::error::Error>> {
        // Train a small network and evaluate performance
        // For this example, we'll use a simple heuristic
        let complexity_penalty = self.shape.layers.len() as f64 * 0.1;
        let size_penalty = self.shape.layers.iter()
            .map(|l| l.layer_type.get_output_size())
            .sum::<usize>() as f64 * 0.01;
        
        // Simple fitness based on architecture efficiency
        self.fitness = Some(1.0 - complexity_penalty - size_penalty);
        Ok(())
    }
}

impl NetworkPhenotype {
    fn fix_connections(&mut self) {
        // Ensure layer connections are valid
        for i in 1..self.shape.layers.len() {
            let prev_output = self.shape.layers[i-1].layer_type.get_output_size();
            if let LayerType::Dense { input_size, .. } = &mut self.shape.layers[i].layer_type {
                *input_size = prev_output;
            }
        }
    }
}
```

### Running Genetic Algorithm

```rust
fn genetic_algorithm_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    let config = NetworkConfig {
        input_size: 4,
        output_size: 3,
        max_hidden_layers: 3,
        max_layer_size: 32,
    };
    
    let options = EvolutionOptions {
        population_size: 20,
        generations: 10,
        mutation_rate: 0.2,
        crossover_rate: 0.8,
        elite_count: 2,
        tournament_size: 3,
    };
    
    println!("Starting genetic algorithm evolution...");
    println!("Population size: {}", options.population_size);
    println!("Generations: {}", options.generations);
    
    // Create initial population
    let mut population: Vec<NetworkPhenotype> = (0..options.population_size)
        .map(|_| NetworkPhenotype::random(&config))
        .collect();
    
    // Evolution loop
    for generation in 0..options.generations {
        // Calculate fitness for all individuals
        for individual in &mut population {
            individual.calculate_fitness(&())?;
        }
        
        // Sort by fitness (descending)
        population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        
        println!("Generation {}: Best fitness = {:.4}, Avg layers = {:.1}", 
                generation,
                population[0].fitness(),
                population.iter().map(|p| p.shape.layers.len()).sum::<usize>() as f64 / population.len() as f64);
        
        // Create next generation
        let mut next_population = Vec::new();
        
        // Keep elite individuals
        next_population.extend(population[..options.elite_count].iter().cloned());
        
        // Generate offspring
        while next_population.len() < options.population_size {
            // Tournament selection
            let parent1 = tournament_select(&population, options.tournament_size);
            let parent2 = tournament_select(&population, options.tournament_size);
            
            let mut offspring = parent1.crossover(parent2);
            offspring.mutate(options.mutation_rate);
            
            next_population.push(offspring);
        }
        
        population = next_population;
    }
    
    // Display best solution
    let best = &population[0];
    println!("\nBest evolved network:");
    for (i, layer) in best.shape.layers.iter().enumerate() {
        println!("Layer {}: {} -> {}, activation: {:?}",
                 i,
                 layer.layer_type.get_input_size(),
                 layer.layer_type.get_output_size(),
                 layer.activation);
    }
    
    Ok(())
}

fn tournament_select(population: &[NetworkPhenotype], tournament_size: usize) -> &NetworkPhenotype {
    use evol::rng::RandomNumberGenerator;
    let mut rng = RandomNumberGenerator::new();
    
    let mut best_idx = rng.fetch_uniform(0.0, population.len() as f64) as usize;
    let mut best_fitness = population[best_idx].fitness();
    
    for _ in 1..tournament_size {
        let idx = rng.fetch_uniform(0.0, population.len() as f64) as usize;
        let fitness = population[idx].fitness();
        if fitness > best_fitness {
            best_idx = idx;
            best_fitness = fitness;
        }
    }
    
    &population[best_idx]
}
```

## Advanced Features

### Memory Management

```rust
use neural::utilities::util::{Utils, WrappedUtils};
use matrix::mat::WrappedMatrix;

fn memory_management_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    // Create utilities with memory limit
    let utils = Utils::new(512 * 1024, 2); // 512KB limit, 2 threads
    let mut wrapped_utils = WrappedUtils::new(utils);
    
    println!("Initial allocated size: {}", wrapped_utils.get_max_allocated_size());
    
    // Create some matrices
    let matrix1 = WrappedMatrix::<f64>::new(100, 100);
    let matrix2 = WrappedMatrix::<f64>::new(50, 200);
    
    // Simulate allocation tracking
    println!("Created matrices for processing");
    
    // The utilities can track memory usage and manage allocation
    // This is useful for large-scale training where memory management is critical
    
    Ok(())
}
```

### Parallel Processing

```rust
use rayon::prelude::*;
use matrix::mat::WrappedMatrix;

fn parallel_processing_tutorial() {
    let matrices: Vec<WrappedMatrix<f64>> = (0..10)
        .map(|_| WrappedMatrix::new(100, 100))
        .collect();
    
    // Process matrices in parallel
    let results: Vec<f64> = matrices.par_iter()
        .map(|matrix| {
            // Simulate some computation
            let mut sum = 0.0;
            for i in 0..matrix.rows() {
                for j in 0..matrix.cols() {
                    sum += matrix.get_unchecked(i, j);
                }
            }
            sum
        })
        .collect();
    
    println!("Processed {} matrices in parallel", results.len());
    println!("Results: {:?}", &results[..3]); // Show first 3 results
}
```

## Performance Optimization

### Benchmarking Your Code

```rust
// Add to Cargo.toml:
// [dev-dependencies]
// criterion = "0.5"

#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    use matrix::mat::WrappedMatrix;
    
    fn benchmark_matrix_operations(c: &mut Criterion) {
        let matrix = WrappedMatrix::<f64>::new(1000, 1000);
        
        // Initialize matrix
        for i in 0..1000 {
            for j in 0..1000 {
                matrix.set_mut_unchecked(i, j, (i * j) as f64);
            }
        }
        
        c.bench_function("matrix_sum", |b| {
            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..1000 {
                    for j in 0..1000 {
                        sum += black_box(matrix.get_unchecked(i, j));
                    }
                }
                black_box(sum)
            })
        });
    }
    
    criterion_group!(benches, benchmark_matrix_operations);
    criterion_main!(benches);
}
```

### Memory-Efficient Training

```rust
fn memory_efficient_training_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    use neural::training::training_params::TrainingParams;
    
    let params = TrainingParams {
        epochs: 1000,
        learning_rate: 0.01,
        batch_size: 16, // Smaller batch size for memory efficiency
        max_cpu_percentage: 70.0, // Limit CPU usage
        print_every: 100,
        ..Default::default()
    };
    
    // Use memory-mapped files for large datasets
    // Process data in batches
    // Use the allocation manager to track memory usage
    
    println!("Memory-efficient training configuration:");
    println!("Batch size: {}", params.batch_size);
    println!("CPU limit: {}%", params.max_cpu_percentage);
    
    Ok(())
}
```

## Real-World Examples

### Iris Classification

Let's implement a complete iris classification example:

```rust
use std::fs::File;
use std::io::{Write, BufWriter};
use neural::training::training_session::TrainingSession;
use neural::training::training_params::TrainingParams;

fn create_iris_dataset() -> Result<(), Box<dyn std::error::Error>> {
    // Sample iris data (in practice, you'd load from a real dataset)
    let iris_data = vec![
        // Setosa
        ([5.1, 3.5, 1.4, 0.2], [1.0, 0.0, 0.0]),
        ([4.9, 3.0, 1.4, 0.2], [1.0, 0.0, 0.0]),
        ([4.7, 3.2, 1.3, 0.2], [1.0, 0.0, 0.0]),
        // Versicolor  
        ([7.0, 3.2, 4.7, 1.4], [0.0, 1.0, 0.0]),
        ([6.4, 3.2, 4.5, 1.5], [0.0, 1.0, 0.0]),
        ([6.9, 3.1, 4.9, 1.5], [0.0, 1.0, 0.0]),
        // Virginica
        ([6.3, 3.3, 6.0, 2.5], [0.0, 0.0, 1.0]),
        ([5.8, 2.7, 5.1, 1.9], [0.0, 0.0, 1.0]),
        ([7.1, 3.0, 5.9, 2.1], [0.0, 0.0, 1.0]),
    ];
    
    // Write input data
    let input_file = File::create("iris_inputs.csv")?;
    let mut input_writer = BufWriter::new(input_file);
    
    for (inputs, _) in &iris_data {
        writeln!(input_writer, "{},{},{},{}", inputs[0], inputs[1], inputs[2], inputs[3])?;
    }
    
    // Write target data
    let target_file = File::create("iris_targets.csv")?;
    let mut target_writer = BufWriter::new(target_file);
    
    for (_, targets) in &iris_data {
        writeln!(target_writer, "{},{},{}", targets[0], targets[1], targets[2])?;
    }
    
    println!("Iris dataset created");
    Ok(())
}

fn create_iris_network_config() -> Result<(), Box<dyn std::error::Error>> {
    let config = r#"
layers:
- layer_type: !Dense
    input_size: 4
    output_size: 8
  activation: ReLU
- layer_type: !Dense
    input_size: 8
    output_size: 6
  activation: ReLU
- layer_type: !Dense
    input_size: 6
    output_size: 3
  activation: Sigmoid
"#;
    
    std::fs::write("iris_network.yaml", config)?;
    println!("Iris network configuration created");
    Ok(())
}

fn train_iris_classifier() -> Result<(), Box<dyn std::error::Error>> {
    let params = TrainingParams {
        epochs: 1000,
        learning_rate: 0.01,
        print_every: 100,
        max_cpu_percentage: 80.0,
        ..Default::default()
    };
    
    let mut session = TrainingSession::new(
        Path::new("./iris_model"),
        Some(Path::new("iris_network.yaml")),
        Path::new("iris_inputs.csv"),
        Path::new("iris_targets.csv"),
        params,
    )?;
    
    println!("Training iris classifier...");
    session.train()?;
    println!("Iris classifier training completed!");
    
    Ok(())
}
```

### Time Series Prediction

```rust
fn create_time_series_data() -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;
    
    let mut input_file = File::create("timeseries_inputs.csv")?;
    let mut target_file = File::create("timeseries_targets.csv")?;
    
    // Generate sine wave data with noise
    for i in 0..1000 {
        let t = i as f64 * 0.1;
        let value = (2.0 * PI * t / 10.0).sin() + 0.1 * (t * 1.3).sin();
        let noise = (i as f64 * 0.137).sin() * 0.05; // Small noise
        let noisy_value = value + noise;
        
        // Use 5 previous values to predict next value
        if i >= 5 {
            let inputs: Vec<f64> = (i-5..i).map(|j| {
                let tj = j as f64 * 0.1;
                (2.0 * PI * tj / 10.0).sin() + 0.1 * (tj * 1.3).sin()
            }).collect();
            
            writeln!(input_file, "{},{},{},{},{}", 
                     inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])?;
            writeln!(target_file, "{}", noisy_value)?;
        }
    }
    
    println!("Time series dataset created");
    Ok(())
}
```

### Model Comparison

```rust
fn model_comparison_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    // Define different network architectures
    let architectures = vec![
        ("small", "layers:\n- layer_type: !Dense\n    input_size: 4\n    output_size: 3\n  activation: Sigmoid"),
        ("medium", "layers:\n- layer_type: !Dense\n    input_size: 4\n    output_size: 8\n  activation: ReLU\n- layer_type: !Dense\n    input_size: 8\n    output_size: 3\n  activation: Sigmoid"),
        ("large", "layers:\n- layer_type: !Dense\n    input_size: 4\n    output_size: 16\n  activation: ReLU\n- layer_type: !Dense\n    input_size: 16\n    output_size: 8\n  activation: ReLU\n- layer_type: !Dense\n    input_size: 8\n    output_size: 3\n  activation: Sigmoid"),
    ];
    
    for (name, config) in architectures {
        println!("Training {} network...", name);
        
        // Save config
        let config_path = format!("{}_network.yaml", name);
        std::fs::write(&config_path, config)?;
        
        // Train model
        let params = TrainingParams {
            epochs: 500,
            learning_rate: 0.01,
            print_every: 100,
            ..Default::default()
        };
        
        let model_dir = format!("./{}_model", name);
        let mut session = TrainingSession::new(
            Path::new(&model_dir),
            Some(Path::new(&config_path)),
            Path::new("iris_inputs.csv"),
            Path::new("iris_targets.csv"),
            params,
        )?;
        
        session.train()?;
        println!("{} network trained and saved to {}", name, model_dir);
    }
    
    println!("All models trained! Compare their performance.");
    Ok(())
}
```

## Best Practices and Tips

### 1. Data Preprocessing

```rust
fn preprocess_data(raw_data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Normalize data to [0, 1] range
    let mut min_vals = vec![f64::INFINITY; raw_data[0].len()];
    let mut max_vals = vec![f64::NEG_INFINITY; raw_data[0].len()];
    
    // Find min/max values
    for row in raw_data {
        for (i, &val) in row.iter().enumerate() {
            min_vals[i] = min_vals[i].min(val);
            max_vals[i] = max_vals[i].max(val);
        }
    }
    
    // Normalize
    raw_data.iter().map(|row| {
        row.iter().enumerate().map(|(i, &val)| {
            if max_vals[i] != min_vals[i] {
                (val - min_vals[i]) / (max_vals[i] - min_vals[i])
            } else {
                0.0
            }
        }).collect()
    }).collect()
}
```

### 2. Error Handling

```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
enum MLError {
    DataLoadError(String),
    TrainingError(String),
    ModelError(String),
}

impl fmt::Display for MLError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MLError::DataLoadError(msg) => write!(f, "Data loading error: {}", msg),
            MLError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            MLError::ModelError(msg) => write!(f, "Model error: {}", msg),
        }
    }
}

impl Error for MLError {}

fn robust_training() -> Result<(), MLError> {
    // Always validate inputs
    if !Path::new("inputs.csv").exists() {
        return Err(MLError::DataLoadError("Input file not found".to_string()));
    }
    
    // Use proper error handling
    match train_model() {
        Ok(_) => println!("Training successful"),
        Err(e) => return Err(MLError::TrainingError(format!("Training failed: {}", e))),
    }
    
    Ok(())
}

fn train_model() -> Result<(), Box<dyn Error>> {
    // Implementation here
    Ok(())
}
```

### 3. Configuration Management

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct ExperimentConfig {
    pub data: DataConfig,
    pub network: NetworkConfigStruct,
    pub training: TrainingConfig,
}

#[derive(Serialize, Deserialize, Debug)]
struct DataConfig {
    pub input_file: String,
    pub target_file: String,
    pub validation_split: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct NetworkConfigStruct {
    pub architecture_file: String,
    pub initialization: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub patience: usize,
}

fn load_config() -> Result<ExperimentConfig, Box<dyn Error>> {
    let config_str = std::fs::read_to_string("experiment_config.yaml")?;
    let config: ExperimentConfig = serde_yaml::from_str(&config_str)?;
    Ok(config)
}
```

### 4. Logging and Monitoring

```rust
fn setup_logging() {
    // In a real application, you'd use the tracing crate
    println!("Setting up experiment logging...");
}

fn log_training_metrics(epoch: usize, loss: f64, accuracy: f64) {
    println!("Epoch {}: Loss = {:.6}, Accuracy = {:.3}%", epoch, loss, accuracy * 100.0);
    
    // Save to file for later analysis
    use std::fs::OpenOptions;
    use std::io::Write;
    
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("training_log.csv")
        .unwrap();
    
    writeln!(file, "{},{},{}", epoch, loss, accuracy).unwrap();
}
```

## Conclusion

This tutorial covered the essential aspects of using the ml_rust library:

1. **Basic Operations**: Matrix manipulation and neural network creation
2. **Training**: Setting up and running training sessions
3. **Genetic Algorithms**: Automated architecture search
4. **Advanced Features**: Memory management and parallel processing
5. **Real-World Examples**: Complete end-to-end projects

### Next Steps

1. **Experiment**: Try different network architectures and hyperparameters
2. **Optimize**: Use benchmarking to identify performance bottlenecks
3. **Scale**: Apply techniques to larger datasets and more complex problems
4. **Contribute**: Add new features or improve existing ones

### Additional Resources

- [API Reference](API.md) - Complete API documentation
- [Architecture Guide](ARCHITECTURE.md) - System design details
- [Development Guide](DEVELOPMENT.md) - Contributing guidelines
- [GitHub Copilot Instructions](.github/copilot-instructions.md) - AI-assisted development

Remember to always validate your results, use proper error handling, and consider the computational resources required for your experiments. The ml_rust library is designed to be both powerful and efficient, but proper usage is key to getting the best results.

Happy machine learning with Rust! 🦀🤖