# Architecture Overview

## System Design

The ml_rust project is designed as a modular machine learning library with a focus on automated neural network architecture search using genetic algorithms. The system is built as a Cargo workspace with multiple specialized crates.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│  train    │  evaluate  │  predict   │  nn_generator         │
├─────────────────────────────────────────────────────────────┤
│                    Core Libraries                           │
├─────────────────────────────────────────────────────────────┤
│  neural   │   evol     │   matrix   │    regret            │
├─────────────────────────────────────────────────────────────┤
│                  Foundation Layer                           │
├─────────────────────────────────────────────────────────────┤
│  alloc    │   utils    │   markov   │    gen               │
└─────────────────────────────────────────────────────────────┘
```

## Crate Responsibilities

### Application Layer

#### `app` - Command Line Interface
- **Purpose**: Provides executable binaries for end-users
- **Components**:
  - `train`: Train neural networks with given data
  - `evaluate`: Evaluate trained networks on test data
  - `predict`: Make predictions with trained networks
  - `nn_generator`: Generate network architectures using genetic algorithms
- **Dependencies**: All other crates as needed
- **Key Features**:
  - Command-line argument parsing with `clap`
  - File I/O and data validation
  - Progress reporting and error handling

### Core Libraries

#### `neural` - Neural Network Implementation
- **Purpose**: Core neural network functionality
- **Components**:
  - `layer`: Layer implementations (Dense, Activation, etc.)
  - `nn`: Complete neural network structures
  - `training`: Training algorithms and session management
  - `activation`: Activation functions (ReLU, Sigmoid, Tanh, etc.)
  - `utilities`: Neural network-specific utilities
- **Key Features**:
  - Backpropagation algorithm
  - Model serialization/deserialization
  - Memory-efficient training
  - Thread-safe operations

#### `evol` - Evolutionary Algorithms
- **Purpose**: Genetic algorithms for automated architecture search
- **Components**:
  - `phenotype`: Trait definitions for evolutionary subjects
  - `evolution`: Evolution engine and strategies
  - `rng`: Random number generation utilities
- **Key Features**:
  - Generic phenotype system
  - Configurable evolution strategies
  - Population management
  - Fitness evaluation

#### `matrix` - Linear Algebra
- **Purpose**: High-performance matrix operations
- **Components**:
  - `mat`: Core matrix implementation
  - `sum_mat`: Specialized sum matrices for probabilistic operations
- **Key Features**:
  - Thread-safe matrix operations
  - Memory-efficient implementations
  - SIMD optimizations (where available)
  - Custom allocator integration

#### `regret` - Regret Minimization
- **Purpose**: Algorithms for minimizing regret in decision-making
- **Components**:
  - Regret minimization strategies
  - Game theory implementations
  - Decision tree algorithms
- **Key Features**:
  - Multi-armed bandit algorithms
  - Counterfactual regret minimization
  - Strategy optimization

### Foundation Layer

#### `alloc` - Memory Management
- **Purpose**: Custom memory allocation strategies
- **Components**:
  - `alloc_manager`: Memory pool management
  - `allocatable`: Trait for allocatable objects
- **Key Features**:
  - Memory pooling for performance
  - Allocation tracking and limits
  - Thread-safe allocation
  - Memory leak prevention

#### `utils` - Common Utilities
- **Purpose**: Shared utility functions
- **Components**:
  - `safer`: Safe abstractions for unsafe operations
  - Error handling helpers
  - Type conversion utilities
- **Key Features**:
  - Poison-resistant mutex operations
  - Safe numeric conversions
  - Common validation functions

#### `gen` - Generation Utilities
- **Purpose**: Network generation and factory patterns
- **Components**:
  - Network builders
  - Configuration generators
  - Template systems
- **Key Features**:
  - Automated network construction
  - Configuration validation
  - Template-based generation

#### `markov` - Markov Chain Implementation
- **Purpose**: Markov chain algorithms and utilities
- **Components**:
  - State transition matrices
  - Markov chain simulation
  - Probability calculations
- **Key Features**:
  - Efficient state representation
  - Transition probability calculation
  - Chain analysis tools

## Data Flow Architecture

### Training Pipeline

```
Input Data (CSV) → Data Validation → Matrix Conversion → Neural Network → Training Loop
                                                              ↓
Output Model ← Model Serialization ← Trained Weights ← Backpropagation
```

### Genetic Algorithm Pipeline

```
Initial Population → Fitness Evaluation → Selection → Crossover/Mutation → New Generation
       ↑                                                                          ↓
Population Restart ← Convergence Check ← Elite Preservation ← Population Update
```

### Memory Management Pipeline

```
Request → Allocation Manager → Memory Pool → Allocate/Deallocate → Usage Tracking
              ↓                                                           ↓
         Size Validation → Pool Management → Memory Limits → Cleanup/GC
```

## Concurrency Model

### Thread Safety Strategy

1. **Immutable Data**: Prefer immutable data structures where possible
2. **Message Passing**: Use channels for communication between threads
3. **Shared State**: Use `Arc<Mutex<T>>` for shared mutable state
4. **Lock-Free**: Use atomic operations for simple shared data

### Parallel Processing

- **Training**: Batch processing with thread pools
- **Matrix Operations**: SIMD and parallel algorithms
- **Genetic Algorithms**: Parallel fitness evaluation
- **I/O Operations**: Async file operations where beneficial

## Error Handling Strategy

### Error Types

1. **Library Errors**: Use `thiserror` for structured error types
2. **Application Errors**: Use `anyhow` for error context
3. **Validation Errors**: Custom validation error types
4. **System Errors**: Wrap standard library errors

### Error Propagation

```rust
// Library level - structured errors
#[derive(Error, Debug)]
pub enum NeuralNetworkError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("Training failed: {0}")]
    TrainingError(String),
}

// Application level - contextual errors
fn train_model() -> anyhow::Result<()> {
    let network = create_network()
        .context("Failed to create neural network")?;
    
    network.train(data)
        .context("Training process failed")?;
    
    Ok(())
}
```

## Performance Considerations

### Memory Management

- **Custom Allocators**: Use memory pools for frequent allocations
- **Zero-Copy**: Minimize data copying where possible
- **Cache Efficiency**: Optimize data layouts for cache performance
- **Memory Limits**: Enforce memory usage limits to prevent OOM

### Computational Efficiency

- **SIMD**: Vectorized operations for numerical computations
- **Parallelism**: Multi-threaded processing for CPU-bound tasks
- **Lazy Evaluation**: Defer expensive computations until needed
- **Algorithmic Optimization**: Choose optimal algorithms for each use case

### I/O Optimization

- **Buffered I/O**: Use buffered readers/writers for file operations
- **Memory Mapping**: Map large files into memory when appropriate
- **Compression**: Compress model files and large datasets
- **Streaming**: Stream large datasets rather than loading entirely

## Testing Strategy

### Test Categories

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test interactions between modules
3. **Property Tests**: Use property-based testing for mathematical functions
4. **Benchmark Tests**: Performance regression testing
5. **End-to-End Tests**: Full workflow testing

### Test Organization

```
crate/
├── src/
│   ├── lib.rs              # Unit tests in modules
│   └── module.rs           # #[cfg(test)] mod tests
├── tests/
│   ├── integration.rs      # Integration tests
│   └── common/
│       └── mod.rs          # Test utilities
└── benches/
    └── performance.rs      # Benchmark tests
```

## Configuration Management

### Configuration Sources

1. **YAML Files**: Network shapes and training parameters
2. **Environment Variables**: Runtime configuration
3. **Command Line**: Override specific parameters
4. **Defaults**: Sensible defaults for all parameters

### Configuration Hierarchy

```
Command Line Args > Environment Variables > Config Files > Defaults
```

## Deployment Considerations

### Build Profiles

- **Development**: Fast compilation, debug symbols
- **Release**: Optimized performance, minimal binary size
- **Benchmarks**: Maximum optimization for performance testing

### Platform Support

- **Primary**: Linux x86_64 (Ubuntu, CentOS)
- **Secondary**: macOS (Intel/Apple Silicon)
- **Future**: Windows, ARM architectures

### Dependencies

- **Minimal**: Keep dependency tree small and auditable
- **Stable**: Prefer stable, well-maintained crates
- **Performance**: Choose dependencies optimized for performance
- **Security**: Regular dependency audits and updates

## Future Architecture Considerations

### Extensibility Points

1. **Layer Types**: Plugin system for custom layer implementations
2. **Optimizers**: Configurable optimization algorithms
3. **Data Loaders**: Pluggable data loading strategies
4. **Metrics**: Custom evaluation metrics

### Scalability

1. **Distributed Training**: Multi-machine training support
2. **GPU Acceleration**: CUDA/OpenCL integration
3. **Cloud Integration**: Cloud storage and compute integration
4. **Streaming**: Real-time data processing capabilities

### Maintenance

1. **Monitoring**: Performance and error monitoring
2. **Logging**: Structured logging for debugging
3. **Profiling**: Built-in profiling capabilities
4. **Documentation**: Auto-generated documentation from code