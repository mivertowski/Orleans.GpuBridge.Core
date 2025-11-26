# Orleans.GpuBridge Testing Framework

Comprehensive testing infrastructure for Orleans.GpuBridge.Core RC2 tests, providing robust utilities for GPU testing with both real hardware and mock providers.

## Overview

This testing framework provides four core components:

1. **GpuTestFixture** - Orleans TestCluster integration with GPU Bridge
2. **MockGpuProvider** - Hardware-independent GPU backend simulation
3. **TestDataBuilders** - Fluent test data generation utilities
4. **GpuTestHelpers** - Assertion helpers, retry logic, and performance measurement

## Features

- ✅ **Orleans TestCluster Integration** - Seamless GPU Bridge setup in test environments
- ✅ **Mock GPU Provider** - Test without hardware dependencies
- ✅ **Fluent Test Data Builders** - Generate arrays, matrices, and edge cases
- ✅ **GPU-Specific Assertions** - Approximate equality, finite values, sorted arrays
- ✅ **Retry Logic** - Handle flaky GPU operations gracefully
- ✅ **Performance Measurement** - Benchmark and profile GPU operations
- ✅ **Thread-Safe** - All components support concurrent testing
- ✅ **Production-Quality** - Comprehensive XML documentation and error handling

## Quick Start

### 1. Basic Test Setup

```csharp
using Orleans.GpuBridge.Tests.RC2.TestingFramework;

public class MyGpuTests : IClassFixture<GpuTestFixture>
{
    private readonly GpuTestFixture _fixture;

    public MyGpuTests(GpuTestFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task MyTest()
    {
        // Access Orleans and GPU Bridge
        var gpuBridge = _fixture.GpuBridge;
        var grainFactory = _fixture.GrainFactory;

        // Your test code here
    }
}
```

### 2. Using Mock GPU Provider

```csharp
[Fact]
public async Task TestWithMockProvider()
{
    // Configure mock behavior
    _fixture.ConfigureMockProvider(mock =>
    {
        mock.SimulateFailure = false;
        mock.SimulatedDelay = TimeSpan.FromMilliseconds(10);
        mock.TrackMetrics = true;
    });

    // Execute operations
    var kernel = await _fixture.GpuBridge.GetKernelAsync<float[], float[]>(
        new KernelId("test-kernel"));

    // Verify mock provider tracked the operation
    _fixture.MockProvider.ExecutionCount.Should().BeGreaterThan(0);
}
```

### 3. Generating Test Data

```csharp
// Float arrays with various patterns
var sequential = TestDataBuilders.FloatArray(1000)
    .WithSequentialValues()
    .Build();

var random = TestDataBuilders.FloatArray(1000)
    .WithRandomRange(-100f, 100f)
    .WithSeed(42)  // Reproducible
    .Build();

var constant = TestDataBuilders.FloatArray(1000)
    .WithConstantValue(5.0f)
    .Build();

// Edge cases
var empty = TestDataBuilders.EdgeCases.EmptyFloatArray();
var sparse = TestDataBuilders.EdgeCases.SparseArray(10000, nonZeroCount: 10);
var maxValues = TestDataBuilders.EdgeCases.MaxFloatValues(100);

// Matrices
var identity = TestDataBuilders.Matrices.IdentityMatrix(4);
var random2D = TestDataBuilders.Matrices.RandomMatrix(10, 10, seed: 42);
var flattened = TestDataBuilders.Matrices.Flatten(random2D);
```

### 4. GPU-Specific Assertions

```csharp
// Approximate equality with tolerance
actual.ShouldBeApproximately(expected, tolerance: 1e-6f);

// Verify finite values (no NaN or Infinity)
result.ShouldContainOnlyFiniteValues();

// Verify sorted arrays
sorted.ShouldBeSorted();

// Verify sum equality (order-independent)
actual.ShouldHaveSameSum(expected, tolerance: 1e-5f);
```

### 5. Retry Logic for Flaky Operations

```csharp
// Retry with default settings (3 retries, 100ms delay)
await GpuTestHelpers.WithRetryAsync(async () =>
{
    await gpuOperation();
});

// Custom retry configuration
await GpuTestHelpers.WithRetryAsync(
    async () => await flakyGpuOperation(),
    maxRetries: 5,
    delayBetweenRetries: TimeSpan.FromMilliseconds(200),
    onRetry: (attempt, ex) =>
    {
        // Log retry attempts
        Console.WriteLine($"Retry {attempt}: {ex.Message}");
    });
```

### 6. Performance Measurement

```csharp
// Simple timing
var elapsed = await GpuTestHelpers.MeasureAsync(async () =>
{
    await gpuOperation();
});

// With result capture
var (result, elapsed) = await GpuTestHelpers.MeasureAsync(async () =>
{
    return await computeIntensiveOperation();
});

// Comprehensive benchmark
var benchmark = await GpuTestHelpers.BenchmarkAsync(
    async () => await kernelExecution(),
    iterations: 100,
    warmupIterations: 10);

Console.WriteLine(benchmark); // Shows min, max, avg, median, P95, P99
```

## Component Details

### GpuTestFixture

Base test fixture that manages Orleans TestCluster lifecycle with GPU Bridge configuration.

**Key Properties:**
- `Cluster` - Orleans TestCluster instance
- `GrainFactory` - Grain factory for creating grain references
- `ServiceProvider` - DI container for services
- `GpuBridge` - IGpuBridge interface
- `MockProvider` - Mock GPU provider (when enabled)

**Configuration:**
```csharp
_fixture.UseMockProvider = true;  // Use mock instead of real hardware
_fixture.EnableVerboseLogging = false;  // Control log output
_fixture.SiloCount = 1;  // Number of silos in cluster
```

**Methods:**
- `InitializeAsync()` - Sets up test cluster
- `DisposeAsync()` - Cleans up resources
- `ResetMockProvider()` - Resets mock to initial state
- `ConfigureMockProvider(action)` - Configure mock behavior
- `WaitForConditionAsync(condition, timeout)` - Wait for async conditions

### MockGpuProvider

Mock implementation of `IGpuBackendProvider` for hardware-independent testing.

**Key Features:**
- Simulates GPU operations (allocation, execution, DMA)
- Configurable behavior (success, failure, delay)
- Metrics tracking and verification
- Thread-safe for concurrent tests

**Configuration Properties:**
```csharp
mock.SimulateFailure = false;  // Throw exceptions
mock.SimulatedDelay = TimeSpan.FromMilliseconds(50);  // Add delay
mock.FailureProbability = 0.1;  // 10% random failures
mock.TrackMetrics = true;  // Enable metrics collection
```

**Metrics:**
```csharp
mock.AllocationCount  // Total allocations
mock.ExecutionCount  // Total kernel executions
mock.TotalBytesAllocated  // Total memory allocated
mock.GetExecutionCount("kernel-name")  // Per-kernel stats
```

### TestDataBuilders

Fluent API for generating test data with reproducible results.

**Array Builders:**
- `FloatArray(size)` - Float array builder
- `IntArray(size)` - Int array builder
- `DoubleArray(size)` - Double array builder
- `Array<T>(size, generator)` - Generic array builder

**Pattern Methods:**
- `.WithSequentialValues()` - 0, 1, 2, 3, ...
- `.WithRandomValues()` - Random values with seed
- `.WithConstantValue(value)` - All elements same value
- `.WithRandomRange(min, max)` - Random values in range
- `.WithGenerator(func)` - Custom generator function
- `.WithSeed(seed)` - Set random seed for reproducibility

**Edge Cases:**
```csharp
CommonSizes.Empty, Single, Small, Medium, Large, Huge
CommonSizes.PowerOfTwo256, PowerOfTwo1024, PowerOfTwo4096

EdgeCases.EmptyFloatArray()
EdgeCases.AllZeros(size)
EdgeCases.MaxFloatValues(size)
EdgeCases.NaNValues(size)
EdgeCases.SparseArray(size, nonZeroCount)
EdgeCases.AlternatingSign(size)
```

**Matrix Builders:**
```csharp
Matrices.IdentityMatrix(size)
Matrices.SequentialMatrix(rows, cols)
Matrices.RandomMatrix(rows, cols, seed)
Matrices.Flatten(matrix)  // Convert 2D to 1D
```

### GpuTestHelpers

Collection of helper methods for robust GPU testing.

**Assertion Helpers:**
- `ShouldBeApproximately(expected, tolerance)` - Float/double array comparison
- `ShouldContainOnlyFiniteValues()` - Verify no NaN/Infinity
- `ShouldBeSorted()` - Verify array is sorted
- `ShouldHaveSameSum(expected, tolerance)` - Order-independent sum comparison

**Retry Logic:**
- `WithRetryAsync(action, maxRetries, delay, onRetry)` - Retry actions
- `WithRetryAsync<T>(func, maxRetries, delay, onRetry)` - Retry functions

**Performance:**
- `MeasureAsync(action)` - Time execution
- `MeasureAsync<T>(func)` - Time and capture result
- `BenchmarkAsync(action, iterations, warmup)` - Full benchmark

**Utilities:**
- `CreateTestLogger(category, minLevel)` - Logger for tests
- `FormatBytes(bytes)` - Human-readable byte sizes
- `FormatDuration(duration)` - Human-readable durations
- `WithTimeoutAsync(action, timeout)` - Timeout protection
- `CreateTimeoutToken(timeout)` - Cancellation token with timeout

## Advanced Usage

### Testing Real vs Mock GPU

```csharp
public class GpuIntegrationTests : IClassFixture<GpuTestFixture>
{
    private readonly GpuTestFixture _fixture;

    public GpuIntegrationTests(GpuTestFixture fixture)
    {
        _fixture = fixture;

        // Use real GPU if available, otherwise mock
        _fixture.UseMockProvider = !IsGpuAvailable();
    }

    private bool IsGpuAvailable()
    {
        // Check for CUDA/OpenCL availability
        return Environment.GetEnvironmentVariable("GPU_TESTS") == "true";
    }
}
```

### Custom Test Data Generators

```csharp
// Generate custom structured data
var customData = TestDataBuilders.Array(1000, i => new MyDataType
{
    Id = i,
    Value = (float)Math.Sin(i * 0.1),
    Timestamp = DateTime.UtcNow.AddSeconds(i)
}).Build();
```

### Parallel Test Execution

```csharp
[Fact]
public async Task ParallelGpuOperations()
{
    var tasks = Enumerable.Range(0, 10).Select(async i =>
    {
        var data = TestDataBuilders.FloatArray(100)
            .WithSeed(i)
            .WithRandomValues()
            .Build();

        return await ProcessOnGpu(data);
    });

    var results = await Task.WhenAll(tasks);
    results.Should().HaveCount(10);
}
```

### Performance Regression Testing

```csharp
[Fact]
public async Task PerformanceRegressionTest()
{
    var baseline = TimeSpan.FromMilliseconds(100);

    var result = await GpuTestHelpers.BenchmarkAsync(
        async () => await gpuKernel.ExecuteAsync(data),
        iterations: 100);

    // Ensure performance hasn't regressed
    result.Average.Should().BeLessThan(baseline * 1.1); // 10% margin
}
```

## Best Practices

1. **Use Reproducible Seeds** - Always specify seeds for random data generation
2. **Test Edge Cases** - Use `EdgeCases` builders for boundary conditions
3. **Approximate Equality** - Use tolerance-based assertions for float comparisons
4. **Retry Flaky Operations** - GPU operations can be timing-sensitive
5. **Mock by Default** - Enable real GPU only for integration tests
6. **Clean Up Resources** - Fixture handles cleanup, but be mindful of resources
7. **Measure Performance** - Use benchmarking for performance-critical operations
8. **Verify Finite Values** - Check for NaN/Infinity in GPU results

## Thread Safety

All components are designed to be thread-safe:
- ✅ `MockGpuProvider` uses concurrent collections
- ✅ `GpuTestFixture` uses proper locking for disposal
- ✅ `TestDataBuilders` generate independent instances
- ✅ `GpuTestHelpers` are stateless utility methods

## Dependencies

- **xUnit** - Test framework
- **FluentAssertions** - Fluent assertion library
- **Moq** - Mocking framework (for additional mocks)
- **Microsoft.Orleans.TestingHost** - Orleans test cluster
- **Orleans.GpuBridge.Runtime** - GPU Bridge implementation

## Examples

See `ExampleUsageTest.cs` for comprehensive usage examples demonstrating:
- Basic fixture usage
- Test data generation patterns
- Assertion helpers
- Retry logic
- Performance measurement
- Benchmarking
- Matrix operations
- Edge case testing
- Formatting utilities

## Contributing

When extending the testing framework:
1. Add XML documentation to all public APIs
2. Create example usage in `ExampleUsageTest.cs`
3. Ensure thread safety for concurrent scenarios
4. Add appropriate error handling
5. Follow fluent API patterns for builders

## License

Part of Orleans.GpuBridge.Core - see project root for license information.
