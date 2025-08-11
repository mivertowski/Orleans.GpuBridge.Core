# Contributing to Orleans.GpuBridge

We love your input! We want to make contributing to Orleans.GpuBridge as easy and transparent as possible.

## Quick Start

- By contributing you agree to the Developer Certificate of Origin (DCO).
- Submit PRs with small, focused commits and tests.
- License: Apache-2.0 for all contributions.

## Development Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new code
3. Update documentation if APIs change
4. Ensure all tests pass: `dotnet test`
5. Make sure code has no warnings: `dotnet build /p:TreatWarningsAsErrors=true`
6. Submit your pull request!

## Development Setup

```bash
# Prerequisites
- .NET 9.0 SDK
- Docker (optional)
- NVIDIA CUDA Toolkit (optional)

# Build
dotnet restore
dotnet build

# Test
dotnet test

# Run with coverage
dotnet test --collect:"XPlat Code Coverage"
```

## Code Style

- Use `dotnet format` before committing
- Follow .NET coding conventions
- Add XML documentation to public APIs
- Keep methods small and focused

## Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `perf:` Performance improvement
- `test:` Test changes

## Testing Guidelines

- Unit tests for all new features
- Aim for 80%+ code coverage
- Use xUnit, Moq, and FluentAssertions
- Run benchmarks for performance-critical code
