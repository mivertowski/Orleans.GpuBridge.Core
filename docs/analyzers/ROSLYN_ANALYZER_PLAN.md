# Roslyn Analyzer Implementation Plan

## Executive Summary

Create a comprehensive Roslyn analyzer package (`Orleans.GpuBridge.Analyzers`) that provides compile-time validation, warnings, and code fixes for GPU-native actor development. This significantly improves developer experience by catching issues before runtime.

**Timeline**: 1-2 weeks (1 developer)
**Impact**: 60-80% reduction in common development errors

---

## Analyzer Categories

### 1. Correctness Analyzers (Errors)

These catch bugs that will cause runtime failures:

| ID | Rule | Severity | Auto-Fix |
|----|------|----------|----------|
| **OGBA001** | ConfigureAwait(false) in grain context | Error | Yes |
| **OGBA002** | Queue capacity not power of 2 | Error | Yes |
| **OGBA003** | Message size not power of 2 | Error | Yes |
| **OGBA004** | Queue capacity out of range (256-1M) | Error | Yes |
| **OGBA005** | Message size out of range (256-4096) | Error | Yes |
| **OGBA006** | Threads per actor exceeds 1024 | Error | No |
| **OGBA007** | Missing IValidateOptions registration | Warning | Yes |
| **OGBA008** | Synchronous blocking in async grain method | Error | No |

### 2. Performance Analyzers (Warnings)

These suggest optimizations:

| ID | Rule | Severity | Auto-Fix |
|----|------|----------|----------|
| **OGBA101** | Temporal ordering enabled unnecessarily | Info | Yes |
| **OGBA102** | Queue capacity too large (memory waste) | Warning | Yes |
| **OGBA103** | Queue capacity too small (frequent overflow risk) | Warning | Yes |
| **OGBA104** | Multiple small messages (suggest batching) | Info | No |
| **OGBA105** | Large message size (suggest splitting) | Warning | Yes |
| **OGBA106** | Inefficient message serialization detected | Info | No |

### 3. Best Practice Analyzers (Info)

These guide developers toward idiomatic usage:

| ID | Rule | Severity | Auto-Fix |
|----|------|----------|----------|
| **OGBA201** | Actor not using GPU-resident state | Info | No |
| **OGBA202** | Missing telemetry registration | Info | Yes |
| **OGBA203** | Missing health check registration | Info | Yes |
| **OGBA204** | Actor not implementing disposal pattern | Warning | Yes |
| **OGBA205** | HLC timestamp not used in message ordering | Info | No |
| **OGBA206** | GPU memory allocation without pooling | Warning | No |

---

## Implementation Architecture

### Project Structure

```
src/Orleans.GpuBridge.Analyzers/
├── Orleans.GpuBridge.Analyzers.csproj
├── Analyzers/
│   ├── Correctness/
│   │   ├── ConfigureAwaitAnalyzer.cs
│   │   ├── PowerOfTwoAnalyzer.cs
│   │   ├── RangeValidationAnalyzer.cs
│   │   ├── ThreadLimitAnalyzer.cs
│   │   └── SynchronousBlockingAnalyzer.cs
│   ├── Performance/
│   │   ├── TemporalOrderingAnalyzer.cs
│   │   ├── QueueSizeAnalyzer.cs
│   │   ├── MessageSizeAnalyzer.cs
│   │   └── BatchingAnalyzer.cs
│   └── BestPractices/
│       ├── TelemetryAnalyzer.cs
│       ├── HealthCheckAnalyzer.cs
│       ├── DisposalPatternAnalyzer.cs
│       └── GpuMemoryAnalyzer.cs
├── CodeFixes/
│   ├── ConfigureAwaitCodeFixProvider.cs
│   ├── PowerOfTwoCodeFixProvider.cs
│   ├── TemporalOrderingCodeFixProvider.cs
│   └── TelemetryCodeFixProvider.cs
├── Resources/
│   └── AnalyzerResources.resx
└── DiagnosticIds.cs

tests/Orleans.GpuBridge.Analyzers.Tests/
├── Correctness/
│   ├── ConfigureAwaitAnalyzerTests.cs
│   ├── PowerOfTwoAnalyzerTests.cs
│   └── ...
├── Performance/
│   └── ...
├── BestPractices/
│   └── ...
└── Verifiers/
    └── CSharpAnalyzerVerifier.cs
```

---

## Detailed Analyzer Specifications

### OGBA001: ConfigureAwait(false) in Grain Context

**Problem**: Using `ConfigureAwait(false)` in Orleans grains breaks single-threaded execution guarantees and causes deadlocks.

**Detection**:
```csharp
// BAD
public async Task MyGrainMethod()
{
    await SomeAsyncOperation().ConfigureAwait(false); // OGBA001
}

// GOOD
public async Task MyGrainMethod()
{
    await SomeAsyncOperation();
}
```

**Implementation**:
1. Detect `ConfigureAwait(false)` invocations
2. Check if containing type inherits from `Grain` or has `[GrainAttribute]`
3. Report diagnostic if in grain context

**Code Fix**: Remove `.ConfigureAwait(false)`

---

### OGBA002/OGBA003: Power-of-2 Validation

**Problem**: Queue capacity and message size must be powers of 2 for efficient GPU indexing.

**Detection**:
```csharp
// BAD
var config = new GpuNativeActorConfiguration
{
    MessageQueueCapacity = 1000, // OGBA002: Not power of 2
    MessageSize = 300            // OGBA003: Not power of 2
};

// GOOD
var config = new GpuNativeActorConfiguration
{
    MessageQueueCapacity = 1024, // Power of 2
    MessageSize = 256            // Power of 2
};
```

**Implementation**:
1. Detect object initializers for `GpuNativeActorConfiguration`
2. Check `MessageQueueCapacity` and `MessageSize` values
3. Validate: `value > 0 && (value & (value - 1)) == 0`

**Code Fix**: Suggest nearest power of 2 (round up or down)

---

### OGBA004/OGBA005: Range Validation

**Problem**: Values outside supported ranges cause runtime failures.

**Detection**:
```csharp
// BAD
MessageQueueCapacity = 100,      // OGBA004: <256 minimum
MessageQueueCapacity = 2_000_000, // OGBA004: >1M maximum
MessageSize = 128,               // OGBA005: <256 minimum
MessageSize = 8192               // OGBA005: >4096 maximum
```

**Implementation**:
1. Check constant values in configuration
2. Validate ranges: Queue (256-1M), Message (256-4096)
3. Report with helpful message

**Code Fix**: Clamp to valid range

---

### OGBA007: Missing IValidateOptions Registration

**Problem**: Configuration validators not registered in DI container.

**Detection**:
```csharp
// BAD - Validator exists but not registered
services.AddGpuBridge(options => { ... });
// Missing: services.AddSingleton<IValidateOptions<...>, ...>()

// GOOD
services.AddGpuBridge(options => { ... });
services.AddSingleton<IValidateOptions<GpuNativeActorConfiguration>,
    GpuNativeActorConfigurationValidator>();
```

**Implementation**:
1. Find types implementing `IValidateOptions<T>`
2. Check if registered in `IServiceCollection`
3. Report if missing

**Code Fix**: Add registration statement

---

### OGBA008: Synchronous Blocking in Async Context

**Problem**: Blocking calls in async grain methods cause deadlocks.

**Detection**:
```csharp
// BAD
public async Task MyGrainMethod()
{
    var result = SomeAsyncOperation().Result; // OGBA008: Synchronous blocking
    SomeAsyncOperation().Wait();              // OGBA008: Synchronous blocking
}

// GOOD
public async Task MyGrainMethod()
{
    var result = await SomeAsyncOperation();
}
```

**Implementation**:
1. Detect `.Result`, `.Wait()`, `.GetAwaiter().GetResult()` in async methods
2. Check if in grain context
3. Report diagnostic

**No auto-fix**: Requires manual await conversion

---

### OGBA101: Temporal Ordering Overhead

**Problem**: Temporal ordering adds 15% overhead but may not be needed.

**Detection**:
```csharp
// Check usage
var config = new GpuNativeActorConfiguration
{
    EnableTemporalOrdering = true // OGBA101: Consider if needed
};
```

**Message**:
```
Temporal ordering adds ~15% performance overhead.
Only enable if you need causal consistency guarantees.
Disable if actors don't depend on message ordering.
```

**Code Fix**: Add comment documenting why temporal ordering is needed

---

### OGBA102/OGBA103: Queue Size Recommendations

**Problem**: Suboptimal queue sizing.

**Detection**:
```csharp
MessageQueueCapacity = 256,     // OGBA103: Too small, frequent overflow risk
MessageQueueCapacity = 1_048_576 // OGBA102: Too large, wastes GPU memory
```

**Recommendations**:
- < 1,000: Too small for production (suggest 4096+)
- > 100,000: Consider if really needed (suggest 16384)
- Ideal range: 4096-65536 for most workloads

**Code Fix**: Suggest recommended value

---

### OGBA201: Missing Telemetry Registration

**Problem**: Metrics not collected for monitoring.

**Detection**:
```csharp
// BAD - No telemetry registration
services.AddGpuBridge(options => { ... });

// GOOD
services.AddGpuBridge(options => { ... });
services.AddSingleton<RingKernelTelemetry>();
services.AddSingleton<TemporalOrderingTelemetry>();
```

**Code Fix**: Add telemetry registration

---

### OGBA204: Disposal Pattern

**Problem**: GPU resources not properly released.

**Detection**:
```csharp
// BAD - Implements IDisposable but doesn't call base.Dispose()
public class MyGpuActor : GpuNativeActorGrain, IDisposable
{
    public void Dispose()
    {
        // Missing: base.Dispose();
    }
}

// GOOD
public class MyGpuActor : GpuNativeActorGrain
{
    // Don't implement IDisposable, base class handles it
}
```

**Code Fix**: Remove IDisposable or call base.Dispose()

---

## Code Fix Provider Examples

### ConfigureAwaitCodeFixProvider

```csharp
// Converts:
await SomeAsync().ConfigureAwait(false);

// To:
await SomeAsync();
```

**Strategy**: Remove `.ConfigureAwait(false)` invocation

### PowerOfTwoCodeFixProvider

```csharp
// Converts:
MessageQueueCapacity = 1000

// To (offers both):
MessageQueueCapacity = 512  // Round down
MessageQueueCapacity = 1024 // Round up
```

**Strategy**: Calculate nearest powers of 2, offer both options

### TemporalOrderingCodeFixProvider

```csharp
// Converts:
EnableTemporalOrdering = true

// To:
EnableTemporalOrdering = true // Required for causal consistency in [use case]
```

**Strategy**: Add explanatory comment

---

## Testing Strategy

### Unit Tests (Per Analyzer)

**Structure**:
```csharp
[Fact]
public async Task ConfigureAwait_InGrainMethod_ReportsDiagnostic()
{
    var test = @"
        using Orleans;
        using System.Threading.Tasks;

        public class MyGrain : Grain
        {
            public async Task Method()
            {
                await Task.Delay(100).{|OGBA001:ConfigureAwait(false)|};
            }
        }";

    await VerifyAnalyzerAsync(test);
}

[Fact]
public async Task ConfigureAwait_OutsideGrain_NoDiagnostic()
{
    var test = @"
        using System.Threading.Tasks;

        public class NotAGrain
        {
            public async Task Method()
            {
                await Task.Delay(100).ConfigureAwait(false); // OK
            }
        }";

    await VerifyAnalyzerAsync(test);
}
```

### Code Fix Tests

```csharp
[Fact]
public async Task ConfigureAwait_CodeFix_RemovesConfigureAwait()
{
    var before = @"
        await Task.Delay(100).ConfigureAwait(false);";

    var after = @"
        await Task.Delay(100);";

    await VerifyCodeFixAsync(before, after);
}
```

### Integration Tests

Test complete scenarios:
1. Developer writes incorrect code
2. Analyzer shows diagnostic
3. Developer applies code fix
4. Code compiles successfully

---

## NuGet Package Configuration

### Package Metadata

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>netstandard2.0</TargetFramework>
    <IncludeBuildOutput>false</IncludeBuildOutput>
    <DevelopmentDependency>true</DevelopmentDependency>
    <IsPackable>true</IsPackable>
    <PackageId>Orleans.GpuBridge.Analyzers</PackageId>
    <Version>1.0.0</Version>
    <Authors>Orleans.GpuBridge Team</Authors>
    <Description>Roslyn analyzers for GPU-native Orleans actors</Description>
    <PackageTags>orleans;gpu;roslyn;analyzer;codefix</PackageTags>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.CodeAnalysis.CSharp" Version="4.8.0" />
    <PackageReference Include="Microsoft.CodeAnalysis.Analyzers" Version="3.3.4" />
  </ItemGroup>

  <ItemGroup>
    <None Include="$(OutputPath)\$(AssemblyName).dll" Pack="true"
          PackagePath="analyzers/dotnet/cs" Visible="false" />
  </ItemGroup>
</Project>
```

### Usage

```bash
# Install analyzer
dotnet add package Orleans.GpuBridge.Analyzers

# Automatically enabled in all projects that reference it
# Shows diagnostics in IDE (Visual Studio, VS Code, Rider)
# Code fixes available via Quick Actions (Ctrl+.)
```

---

## Documentation

### User Guide

**docs/analyzers/ANALYZER_GUIDE.md**:
- Complete list of analyzer rules
- Examples of each diagnostic
- Available code fixes
- Configuration options (severity levels)

### Developer Guide

**docs/analyzers/ANALYZER_DEVELOPMENT.md**:
- How to create new analyzers
- Testing patterns
- Debugging analyzers
- Performance considerations

---

## Success Metrics

### Developer Experience Improvements

- **Compile-time error detection**: 80% of configuration errors caught before runtime
- **Time savings**: 30-60 minutes saved per developer per week
- **Code quality**: 60% reduction in Orleans grain deadlocks
- **Onboarding**: New developers productive 2× faster

### Adoption Metrics

- Analyzer install rate: Target >90% of GPU actor projects
- Code fix usage: Target >70% of diagnostics auto-fixed
- Developer satisfaction: Target >4.5/5 rating

---

## Implementation Timeline

### Week 1: Core Analyzers

- [x] Project setup
- [ ] OGBA001: ConfigureAwait analyzer + code fix
- [ ] OGBA002/003: Power-of-2 analyzer + code fix
- [ ] OGBA004/005: Range validation analyzer + code fix
- [ ] OGBA008: Synchronous blocking analyzer
- [ ] Unit tests for core analyzers

### Week 2: Advanced Analyzers & Polish

- [ ] OGBA101-106: Performance analyzers
- [ ] OGBA201-206: Best practice analyzers
- [ ] Code fix providers for all auto-fixable rules
- [ ] Integration tests
- [ ] Documentation
- [ ] NuGet package configuration

---

## Risks & Mitigations

### Risk: False Positives

**Mitigation**: Comprehensive test suite, conservative detection patterns

### Risk: Performance Impact

**Mitigation**: Incremental analyzers, caching, performance benchmarks

### Risk: Breaking Changes

**Mitigation**: Semantic versioning, deprecation warnings, migration guide

---

## Future Enhancements (v2.0)

- **OGBA301**: GPU memory leak detection (static analysis)
- **OGBA302**: Ring kernel infinite loop detection
- **OGBA303**: Message queue deadlock detection
- **OGBA304**: Temporal ordering violation detection
- **Code generation**: Auto-generate boilerplate for GPU actors
- **Refactorings**: Extract actor, convert to GPU-native, etc.

---

## Conclusion

A comprehensive Roslyn analyzer will significantly improve developer experience by:
1. Catching errors at compile-time (before runtime)
2. Guiding developers toward best practices
3. Providing instant code fixes for common issues
4. Reducing onboarding time for new developers
5. Improving code quality and reducing bugs

**ROI**: High - minimal implementation cost, significant long-term value

**Next Steps**: Implement core analyzers (Week 1), then advanced features (Week 2)
