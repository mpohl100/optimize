# Development Guide

This guide provides comprehensive instructions for setting up a development environment and contributing to the ml_rust project.

## Prerequisites

### Required Tools

- **Rust**: Version 1.75.0 or later (edition 2021)
- **Git**: For version control
- **Make**: For automation (optional but recommended)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential git curl
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Or install via Homebrew
brew install git
```

#### Windows
- Install [Git for Windows](https://git-scm.com/download/win)
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Rust Installation

Install Rust using rustup:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Install required components:
```bash
rustup component add clippy rustfmt
rustup toolchain install nightly  # For benchmarks
```

## Setting Up the Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/mpohl100/ml_rust.git
cd ml_rust
```

### 2. Build the Project

```bash
# Build all crates
cargo build

# Build with optimizations (for testing performance)
cargo build --release
```

### 3. Run Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific crate tests
cargo test -p neural

# Run with all features
cargo test --all-features
```

### 4. Development Tools Setup

#### VS Code Setup

Install recommended extensions:
```json
{
    "recommendations": [
        "rust-lang.rust-analyzer",
        "vadimcn.vscode-lldb",
        "serayuzgur.crates",
        "tamasfe.even-better-toml",
        "usernamehw.errorlens"
    ]
}
```

#### Editor Configuration

The project includes `.editorconfig` for consistent formatting across editors.

## Code Quality Standards

### Formatting

Always format code before committing:
```bash
cargo fmt --all
```

### Linting

Fix all clippy warnings:
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Testing

Maintain high test coverage:
```bash
# Install coverage tool
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage/
```

## Development Workflow

### 1. Branch Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical fixes for production

### 2. Feature Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-activation-function
   ```

2. **Implement Feature**
   - Write tests first (TDD approach)
   - Implement functionality
   - Add documentation
   - Update examples

3. **Quality Checks**
   ```bash
   # Format code
   cargo fmt --all
   
   # Run lints
   cargo clippy --all-targets --all-features -- -D warnings
   
   # Run tests
   cargo test --all-features
   
   # Check documentation
   cargo doc --no-deps --all-features
   
   # Run benchmarks (if applicable)
   cargo bench
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new activation function with tests and docs"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/new-activation-function
   # Create PR via GitHub interface
   ```

### 3. Commit Message Convention

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `ci:` CI/CD changes

Examples:
```
feat(neural): add batch normalization layer
fix(matrix): resolve memory leak in multiplication
docs(readme): update installation instructions
test(evol): add property-based tests for crossover
```

## Testing Guidelines

### Unit Testing

Write unit tests for all public functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_relu() {
        let activation = ActivationFunction::ReLU;
        
        assert_eq!(activation.apply(2.0), 2.0);
        assert_eq!(activation.apply(-1.0), 0.0);
        assert_eq!(activation.apply(0.0), 0.0);
    }

    #[test]
    fn test_activation_derivative() {
        let activation = ActivationFunction::ReLU;
        
        assert_eq!(activation.derivative(2.0), 1.0);
        assert_eq!(activation.derivative(-1.0), 0.0);
    }
}
```

### Integration Testing

Create integration tests in `tests/` directory:

```rust
// tests/neural_network_integration.rs
use neural::*;

#[test]
fn test_complete_training_workflow() {
    // Create network
    let mut network = NeuralNetwork::new(vec![
        Layer::Dense { input: 2, output: 4, activation: ReLU },
        Layer::Dense { input: 4, output: 1, activation: Sigmoid },
    ]).unwrap();

    // Load data
    let training_data = load_test_data("tests/data/xor.csv").unwrap();
    
    // Train network
    let result = network.train(&training_data, 1000);
    assert!(result.is_ok());
    
    // Test accuracy
    let accuracy = network.evaluate(&training_data).unwrap();
    assert!(accuracy > 0.9);
}
```

### Property-Based Testing

Use proptest for mathematical functions:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_matrix_multiplication_properties(
        a_rows in 1..10usize,
        shared_dim in 1..10usize,
        b_cols in 1..10usize,
    ) {
        let a = Matrix::random(a_rows, shared_dim);
        let b = Matrix::random(shared_dim, b_cols);
        
        let result = &a * &b;
        
        prop_assert_eq!(result.rows(), a_rows);
        prop_assert_eq!(result.cols(), b_cols);
    }
}
```

### Benchmark Testing

Create benchmarks for performance-critical code:

```rust
// benches/matrix_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use matrix::Matrix;

fn benchmark_matrix_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    
    for size in [64, 128, 256, 512] {
        let a = Matrix::random(size, size);
        let b = Matrix::random(size, size);
        
        group.bench_with_input(
            format!("{}x{}", size, size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a) * black_box(&b);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_matrix_mul);
criterion_main!(benches);
```

## Documentation Standards

### Code Documentation

Document all public APIs:

```rust
/// Applies the ReLU activation function.
///
/// The ReLU (Rectified Linear Unit) function returns the input if positive,
/// otherwise returns zero: `f(x) = max(0, x)`.
///
/// # Arguments
///
/// * `x` - Input value to apply activation to
///
/// # Returns
///
/// The activated value
///
/// # Examples
///
/// ```
/// use neural::activation::ActivationFunction;
///
/// let relu = ActivationFunction::ReLU;
/// assert_eq!(relu.apply(2.0), 2.0);
/// assert_eq!(relu.apply(-1.0), 0.0);
/// ```
///
/// # Performance
///
/// This function has O(1) time complexity and is highly optimized
/// for vectorized operations.
pub fn apply(&self, x: f64) -> f64 {
    match self {
        Self::ReLU => x.max(0.0),
        // ... other variants
    }
}
```

### Module Documentation

Add module-level documentation:

```rust
//! Activation function implementations.
//!
//! This module provides various activation functions commonly used in neural networks:
//! - ReLU and its variants
//! - Sigmoid and Tanh
//! - Softmax for classification
//! - Linear for output layers
//!
//! # Usage
//!
//! ```rust
//! use neural::activation::ActivationFunction;
//!
//! let activation = ActivationFunction::ReLU;
//! let result = activation.apply(2.5);
//! ```
```

### README Updates

Keep README.md current with:
- Installation instructions
- Basic usage examples
- Feature descriptions
- Contributing guidelines

## Performance Optimization

### Profiling

Use built-in profiling tools:

```bash
# Install perf (Linux)
sudo apt install linux-perf

# Profile with perf
cargo build --release
perf record --call-graph dwarf target/release/train --model-dir /tmp/model
perf report
```

Use cargo-flamegraph for flame graphs:

```bash
cargo install flamegraph
cargo flamegraph --bin train -- --model-dir /tmp/model
```

### Memory Profiling

Use valgrind for memory analysis:

```bash
# Install valgrind
sudo apt install valgrind

# Run with memory checking
cargo build
valgrind --tool=memcheck target/debug/train --model-dir /tmp/model
```

### Benchmark Regression Testing

Set up benchmark comparison:

```bash
# Install cargo-criterion
cargo install cargo-criterion

# Run benchmarks and save baseline
cargo criterion --save-baseline main

# After changes, compare
cargo criterion --baseline main
```

## Debugging

### Debug Builds

Use debug builds for development:

```bash
# Debug build with symbols
cargo build

# Run with debugger
rust-gdb target/debug/train
```

### Logging

Use structured logging:

```rust
use tracing::{info, debug, error, instrument};

#[instrument]
pub fn train_network(data: &TrainingData) -> Result<()> {
    info!("Starting training with {} samples", data.len());
    
    for epoch in 0..epochs {
        debug!("Training epoch {}", epoch);
        // ... training logic
    }
    
    info!("Training completed successfully");
    Ok(())
}
```

Enable logging in tests:

```bash
RUST_LOG=debug cargo test
```

### Memory Debugging

Use AddressSanitizer for memory issues:

```bash
# Build with sanitizer
RUSTFLAGS="-Z sanitizer=address" cargo build --target x86_64-unknown-linux-gnu

# Run with memory checking
./target/x86_64-unknown-linux-gnu/debug/train
```

## Continuous Integration

### Local CI Simulation

Run the same checks as CI locally:

```bash
# Format check
cargo fmt --all -- --check

# Clippy check
cargo clippy --all-targets --all-features -- -D warnings

# Test all features
cargo test --all-features

# Build documentation
cargo doc --no-deps --all-features

# Run benchmarks
cargo bench --no-run
```

### Pre-commit Hooks

Set up pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Release Process

### Version Management

Update versions in `Cargo.toml`:

```toml
[package]
version = "0.2.0"  # Increment according to semver
```

### Changelog

Update `CHANGELOG.md` following Keep a Changelog format:

```markdown
## [0.2.0] - 2024-01-15

### Added
- New activation functions (Swish, GELU)
- Batch normalization layers
- Learning rate scheduling

### Changed
- Improved matrix multiplication performance
- Updated dependencies to latest versions

### Fixed
- Memory leak in training loop
- Numerical instability in softmax
```

### Release Checklist

1. [ ] Update version numbers
2. [ ] Update CHANGELOG.md
3. [ ] Run full test suite
4. [ ] Generate documentation
5. [ ] Create release tag
6. [ ] Publish to crates.io (if applicable)

## Contributing Guidelines

### Code Style

- Follow Rust standard conventions
- Use meaningful variable and function names
- Keep functions small and focused
- Add comments for complex algorithms
- Use type annotations where helpful for clarity

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests and documentation
4. Ensure all CI checks pass
5. Submit pull request with clear description
6. Address review feedback
7. Squash commits if requested

### Issue Reporting

When reporting issues:
- Use issue templates
- Provide minimal reproduction case
- Include system information
- Add relevant logs/error messages
- Tag with appropriate labels

## Troubleshooting

### Common Issues

**Build Failures**
```bash
# Clean and rebuild
cargo clean
cargo build

# Update dependencies
cargo update
```

**Test Failures**
```bash
# Run specific test with output
cargo test test_name -- --nocapture

# Run tests sequentially
cargo test -- --test-threads=1
```

**Performance Issues**
```bash
# Profile with release build
cargo build --release
perf record target/release/app
```

### Getting Help

- Check existing issues on GitHub
- Review documentation
- Ask questions in discussions
- Join the Rust community Discord
- Read the Rust Book and documentation

## Development Tools

### Recommended Cargo Extensions

```bash
# Code coverage
cargo install cargo-tarpaulin

# Audit dependencies
cargo install cargo-audit

# Check for outdated dependencies
cargo install cargo-outdated

# Generate flame graphs
cargo install flamegraph

# Benchmark comparison
cargo install cargo-criterion

# Documentation serving
cargo install basic-http-server
```

### IDE/Editor Plugins

**VS Code**
- rust-analyzer: Language server
- CodeLLDB: Debugging support
- Error Lens: Inline error display
- Crates: Dependency management

**Vim/Neovim**
- rust.vim: Syntax highlighting
- coc-rust-analyzer: Language server
- vim-cargo: Cargo integration

**IntelliJ/CLion**
- Rust plugin: Full IDE support
- TOML plugin: Configuration files