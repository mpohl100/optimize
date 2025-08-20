# Test Coverage Workflow

This repository includes an automated test coverage measurement system that runs on every pull request and provides detailed coverage statistics for each Rust crate in the workspace.

## Features

### Automated Coverage Measurement
- **Per-crate Coverage**: Measures test coverage for each individual crate in the workspace
- **Workspace Aggregation**: Provides total coverage statistics across all crates
- **Detailed Breakdown**: Shows lines covered, total lines, and coverage percentage

### Pull Request Integration
- **Automatic Comments**: Posts coverage reports as PR comments
- **Smart Updates**: Updates existing coverage comments instead of creating duplicates
- **Error Handling**: Gracefully handles cases where coverage cannot be generated

### Coverage Tools
- **cargo-tarpaulin**: Uses the industry-standard Rust coverage tool
- **JSON Output**: Processes structured coverage data for accurate reporting
- **Artifact Storage**: Stores coverage reports as GitHub Actions artifacts

## Workflow Triggers

The coverage workflow runs on:
- **Pull Requests**: Every PR targeting the `main` branch
- **Push to Main**: Direct pushes to the main branch

## Coverage Report Format

The workflow generates comments in the following format:

```markdown
## 📊 Test Coverage Report

Coverage Report
==============
| Crate | Lines Covered | Coverage % |
|-------|---------------|------------|
| matrix | 69/580 | 11.89% |
| utils | 0/542 | 0% |
| evol | 15/423 | 3.55% |

**Total Coverage: 84/1545 (5.43%)**

Generated at: 2025-08-20 19:54:04 UTC
```

## Configuration

The workflow is defined in `.github/workflows/coverage.yml` and includes:

- **Rust Toolchain**: Uses stable Rust toolchain
- **Dependencies**: Automatically installs cargo-tarpaulin, bc, and jq
- **Caching**: Caches Cargo dependencies for faster builds
- **Parallel Processing**: Processes each crate individually for better granularity

## Exclusions

The coverage measurement excludes:
- Test files (`*/tests/*`)
- Benchmark files (`*/benches/*`)
- Build artifacts (`target/*`)

## Requirements

For the workflow to function properly:
- Tests must be present in the crates (crates without tests show 0% coverage)
- The workspace must be properly configured with all crates listed in the root `Cargo.toml`

## Troubleshooting

If coverage reports are not generated:
1. Check that the crate has unit tests
2. Verify the crate builds successfully
3. Check the workflow logs for any cargo-tarpaulin errors
4. Ensure all dependencies are properly specified

The workflow includes error handling and will post a notification if coverage generation fails.