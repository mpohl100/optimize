# Per-Crate GitHub Workflows

This repository includes individual GitHub Actions workflows for each crate in the workspace. These workflows provide targeted CI/CD for each crate while maintaining the same quality standards as the main workflow.

## Workflow Files

The following per-crate workflows are available:

- `.github/workflows/build_and_test_alloc.yml` - Memory allocation management
- `.github/workflows/build_and_test_app.yml` - Command line interface applications  
- `.github/workflows/build_and_test_evol.yml` - Evolutionary algorithms
- `.github/workflows/build_and_test_gen.yml` - Generation utilities
- `.github/workflows/build_and_test_markov.yml` - Markov chain implementation
- `.github/workflows/build_and_test_matrix.yml` - Linear algebra operations
- `.github/workflows/build_and_test_neural.yml` - Neural network implementation
- `.github/workflows/build_and_test_regret.yml` - Regret minimization algorithms
- `.github/workflows/build_and_test_utils.yml` - Common utilities

## Triggers

Each workflow is triggered by:

1. **Push to main branch** - When files in the crate's directory or workspace configuration change
2. **Pull requests to main** - When files in the crate's directory or workspace configuration change  
3. **Manual dispatch** - Can be triggered manually from the GitHub Actions UI

## Path Filters

Each workflow only runs when changes are made to:
- Files in the crate's subdirectory (e.g., `alloc/**` for the alloc crate)
- Workspace configuration files (`Cargo.toml`, `Cargo.lock`)

## Build Steps

Each workflow runs the same quality checks as the main workflow, but scoped to the specific crate:

1. `cargo build --release --verbose -p <crate>` - Build the crate in release mode
2. `cargo test --release --verbose -p <crate>` - Run tests for the crate
3. `cargo bench -p <crate>` - Run benchmarks (if available)
4. `cargo fmt --check` - Check code formatting (workspace-wide)
5. `cargo clippy -- -D warnings` - Run linting checks (workspace-wide)

## Development Environment

All workflows use the same devcontainer setup as the main workflow to ensure consistency across different environments.