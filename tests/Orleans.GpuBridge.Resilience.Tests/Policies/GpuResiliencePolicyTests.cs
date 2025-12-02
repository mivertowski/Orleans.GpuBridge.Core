// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Resilience.Policies;

namespace Orleans.GpuBridge.Resilience.Tests.Policies;

/// <summary>
/// Tests for <see cref="GpuResiliencePolicy"/> class.
/// </summary>
public class GpuResiliencePolicyTests : IDisposable
{
    private readonly Mock<ILogger<GpuResiliencePolicy>> _loggerMock;
    private readonly GpuResiliencePolicyOptions _options;
    private readonly GpuResiliencePolicy _policy;

    public GpuResiliencePolicyTests()
    {
        _loggerMock = new Mock<ILogger<GpuResiliencePolicy>>();
        _options = new GpuResiliencePolicyOptions
        {
            RetryOptions =
            {
                MaxAttempts = 3,
                BaseDelay = TimeSpan.FromMilliseconds(10),
                MaxDelay = TimeSpan.FromMilliseconds(100)
            },
            CircuitBreakerOptions =
            {
                FailureRatio = 0.5,
                SamplingDuration = TimeSpan.FromSeconds(10),
                MinimumThroughput = 5,
                BreakDuration = TimeSpan.FromSeconds(5)
            },
            TimeoutOptions =
            {
                KernelExecution = TimeSpan.FromSeconds(30),
                DeviceOperation = TimeSpan.FromSeconds(10),
                MemoryAllocation = TimeSpan.FromSeconds(5),
                KernelCompilation = TimeSpan.FromMinutes(1)
            },
            BulkheadOptions =
            {
                MaxConcurrentOperations = 5
            }
        };

        _policy = new GpuResiliencePolicy(
            _loggerMock.Object,
            Options.Create(_options));
    }

    [Fact]
    public void Constructor_ShouldInitializeSuccessfully()
    {
        // Assert
        _policy.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_ShouldThrowOnNullLogger()
    {
        // Act
        var act = () => new GpuResiliencePolicy(null!, Options.Create(_options));

        // Assert
        act.Should().Throw<ArgumentNullException>().WithParameterName("logger");
    }

    [Fact]
    public void Constructor_ShouldThrowOnNullOptions()
    {
        // Act
        var act = () => new GpuResiliencePolicy(_loggerMock.Object, null!);

        // Assert
        act.Should().Throw<ArgumentNullException>().WithParameterName("options");
    }

    [Fact]
    public async Task ExecuteKernelOperationAsync_ShouldExecuteSuccessfully()
    {
        // Arrange
        var expectedResult = 42;

        // Act
        var result = await _policy.ExecuteKernelOperationAsync(
            async ct =>
            {
                await Task.Delay(1, ct);
                return expectedResult;
            },
            "TestOperation");

        // Assert
        result.Should().Be(expectedResult);
    }

    [Fact]
    public async Task ExecuteKernelOperationAsync_ShouldRespectCancellation()
    {
        // Arrange
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        var act = async () => await _policy.ExecuteKernelOperationAsync(
            async ct =>
            {
                await Task.Delay(1000, ct);
                return 42;
            },
            "TestOperation",
            cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    [Fact]
    public async Task ExecuteDeviceOperationAsync_ShouldExecuteSuccessfully()
    {
        // Arrange
        var expectedResult = "device-result";

        // Act
        var result = await _policy.ExecuteDeviceOperationAsync(
            async ct =>
            {
                await Task.Delay(1, ct);
                return expectedResult;
            },
            "GPU-0",
            "TestDeviceOp");

        // Assert
        result.Should().Be(expectedResult);
    }

    [Fact]
    public async Task ExecuteMemoryAllocationAsync_ShouldExecuteSuccessfully()
    {
        // Arrange
        var expectedPtr = new IntPtr(12345);

        // Act
        var result = await _policy.ExecuteMemoryAllocationAsync(
            async ct =>
            {
                await Task.Delay(1, ct);
                return expectedPtr;
            },
            1024 * 1024);

        // Assert
        result.Should().Be(expectedPtr);
    }

    [Fact]
    public async Task ExecuteCompilationAsync_ShouldExecuteSuccessfully()
    {
        // Arrange
        var expectedModule = "compiled-kernel";

        // Act
        var result = await _policy.ExecuteCompilationAsync(
            async ct =>
            {
                await Task.Delay(1, ct);
                return expectedModule;
            },
            "TestKernel");

        // Assert
        result.Should().Be(expectedModule);
    }

    [Fact]
    public void GetBulkheadMetrics_ShouldReturnValidMetrics()
    {
        // Act
        var metrics = _policy.GetBulkheadMetrics();

        // Assert
        metrics.TotalSlots.Should().Be(_options.BulkheadOptions.MaxConcurrentOperations);
        metrics.AvailableSlots.Should().Be(_options.BulkheadOptions.MaxConcurrentOperations);
        metrics.InUseSlots.Should().Be(0);
        metrics.UtilizationPercentage.Should().Be(0);
    }

    [Fact]
    public async Task GetBulkheadMetrics_ShouldReflectActiveOperations()
    {
        // Arrange
        var tcs = new TaskCompletionSource<bool>();
        var operationStarted = new TaskCompletionSource<bool>();

        // Start an operation that blocks
        var operationTask = Task.Run(async () =>
        {
            await _policy.ExecuteMemoryAllocationAsync(
                async ct =>
                {
                    operationStarted.SetResult(true);
                    await tcs.Task;
                    return IntPtr.Zero;
                },
                1024);
        });

        // Wait for operation to start
        await operationStarted.Task;
        await Task.Delay(50); // Give time for semaphore to be acquired

        // Act
        var metrics = _policy.GetBulkheadMetrics();

        // Assert
        metrics.InUseSlots.Should().Be(1);
        metrics.AvailableSlots.Should().Be(_options.BulkheadOptions.MaxConcurrentOperations - 1);
        metrics.UtilizationPercentage.Should().BeGreaterThan(0);

        // Cleanup
        tcs.SetResult(true);
        await operationTask;
    }

    [Fact]
    public async Task ShutdownAsync_ShouldCompleteSuccessfully()
    {
        // Act
        await _policy.ShutdownAsync(TimeSpan.FromSeconds(1));

        // Assert - no exception thrown
    }

    [Fact]
    public void Dispose_ShouldNotThrow()
    {
        // Act
        var act = () => _policy.Dispose();

        // Assert
        act.Should().NotThrow();
    }

    [Fact]
    public void Dispose_ShouldBeIdempotent()
    {
        // Act
        var act = () =>
        {
            _policy.Dispose();
            _policy.Dispose();
            _policy.Dispose();
        };

        // Assert
        act.Should().NotThrow();
    }

    public void Dispose()
    {
        _policy?.Dispose();
    }
}
