// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Runtime.RingKernels;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.RingKernelTests;

/// <summary>
/// Tests for the CpuFallbackRingKernelBridge implementation.
/// </summary>
/// <remarks>
/// Validates CPU-only execution of the IRingKernelBridge interface
/// for environments without GPU hardware.
/// </remarks>
public class CpuFallbackRingKernelBridgeTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly CpuFallbackRingKernelBridge _bridge;
    private readonly ILogger<CpuFallbackRingKernelBridge> _logger;

    public CpuFallbackRingKernelBridgeTests(ITestOutputHelper output)
    {
        _output = output;

        var services = new ServiceCollection();
        services.AddLogging(builder => builder
            .AddDebug()
            .SetMinimumLevel(LogLevel.Debug));

        var sp = services.BuildServiceProvider();
        _logger = sp.GetRequiredService<ILogger<CpuFallbackRingKernelBridge>>();

        _bridge = new CpuFallbackRingKernelBridge(_logger);

        _output.WriteLine("✅ CpuFallbackRingKernelBridge initialized");
    }

    [Fact]
    public async Task IsGpuAvailableAsync_ShouldReturnFalse()
    {
        // Act
        var isAvailable = await _bridge.IsGpuAvailableAsync();

        // Assert
        Assert.False(isAvailable);
        _output.WriteLine("✅ GPU correctly reported as unavailable for CPU fallback");
    }

    [Fact]
    public async Task GetDevicePlacementAsync_ShouldReturnMinusOne()
    {
        // Arrange
        var actorKey = "test-actor-123";

        // Act
        var deviceId = await _bridge.GetDevicePlacementAsync(actorKey);

        // Assert
        Assert.Equal(-1, deviceId);
        _output.WriteLine($"✅ Device placement correctly returned -1 (no GPU) for actor {actorKey}");
    }

    [Fact]
    public async Task AllocateStateAsync_ShouldReturnValidHandle()
    {
        // Arrange
        var actorId = "test-actor-alloc";
        var deviceId = -1;

        // Act
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, deviceId);

        // Assert
        Assert.NotNull(handle);
        Assert.Equal(actorId, handle.ActorId);
        Assert.Equal(deviceId, handle.DeviceId);
        Assert.True(handle.SizeBytes > 0);
        Assert.Equal(IntPtr.Zero, handle.GpuPointer); // CPU fallback has no GPU pointer

        _output.WriteLine($"✅ State allocated for actor {actorId}");
        _output.WriteLine($"   Size: {handle.SizeBytes} bytes");
        _output.WriteLine($"   GPU pointer: {handle.GpuPointer} (expected Zero for CPU)");

        // Cleanup
        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task ReleaseStateAsync_ShouldSucceed()
    {
        // Arrange
        var actorId = "test-actor-release";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);

        // Act
        await _bridge.ReleaseStateAsync(handle);

        // Assert - verify handle is disposed (no exception thrown)
        _output.WriteLine($"✅ State released successfully for actor {actorId}");
    }

    [Fact]
    public async Task ReadStateAsync_ShouldReturnDefaultState()
    {
        // Arrange
        var actorId = "test-actor-read";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);

        // Act
        var state = await _bridge.ReadStateAsync(handle);

        // Assert
        Assert.Equal(default(TestState), state);

        _output.WriteLine($"✅ Read default state for actor {actorId}");
        _output.WriteLine($"   Value: {state.Value}");
        _output.WriteLine($"   Counter: {state.Counter}");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task WriteStateAsync_ShouldUpdateState()
    {
        // Arrange
        var actorId = "test-actor-write";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);

        var newState = new TestState { Value = 42.5f, Counter = 100 };

        // Act
        await _bridge.WriteStateAsync(handle, newState);
        var readState = await _bridge.ReadStateAsync(handle);

        // Assert
        Assert.Equal(newState.Value, readState.Value);
        Assert.Equal(newState.Counter, readState.Counter);

        _output.WriteLine($"✅ State written and read back successfully");
        _output.WriteLine($"   Value: {readState.Value}");
        _output.WriteLine($"   Counter: {readState.Counter}");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task ExecuteHandlerAsync_ShouldReturnCpuFallbackResult()
    {
        // Arrange
        var actorId = "test-actor-execute";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);
        handle.ShadowState = new TestState { Value = 10.0f, Counter = 5 };

        var request = new TestRequest { Operand = 5.0f };

        // Act
        var result = await _bridge.ExecuteHandlerAsync<TestRequest, TestResponse, TestState>(
            "TestKernel",
            handlerId: 1,
            request,
            handle);

        // Assert
        Assert.True(result.Success);
        Assert.False(result.WasGpuExecution); // CPU fallback
        Assert.True(result.LatencyNs > 0);

        _output.WriteLine($"✅ Handler executed via CPU fallback");
        _output.WriteLine($"   Success: {result.Success}");
        _output.WriteLine($"   Was GPU: {result.WasGpuExecution}");
        _output.WriteLine($"   Latency: {result.LatencyNs} ns");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task ExecuteFireAndForgetAsync_ShouldReturnState()
    {
        // Arrange
        var actorId = "test-actor-fandf";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);
        var initialState = new TestState { Value = 20.0f, Counter = 10 };
        handle.ShadowState = initialState;

        var request = new TestRequest { Operand = 3.0f };

        // Act
        var newState = await _bridge.ExecuteFireAndForgetAsync<TestRequest, TestState>(
            "TestKernel",
            handlerId: 2,
            request,
            handle);

        // Assert
        Assert.Equal(initialState.Value, newState.Value);
        Assert.Equal(initialState.Counter, newState.Counter);

        _output.WriteLine($"✅ Fire-and-forget executed via CPU fallback");
        _output.WriteLine($"   State value: {newState.Value}");
        _output.WriteLine($"   State counter: {newState.Counter}");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public void GetTelemetry_ShouldReturnValidMetrics()
    {
        // Act
        var telemetry = _bridge.GetTelemetry();

        // Assert
        Assert.NotNull(telemetry);
        Assert.Equal(0, telemetry.GpuExecutions); // CPU fallback has no GPU
        Assert.Equal(0, telemetry.AllocatedGpuMemoryBytes);

        _output.WriteLine($"✅ Telemetry retrieved");
        _output.WriteLine($"   Total executions: {telemetry.TotalExecutions}");
        _output.WriteLine($"   GPU executions: {telemetry.GpuExecutions}");
        _output.WriteLine($"   CPU fallbacks: {telemetry.CpuFallbackExecutions}");
        _output.WriteLine($"   Active handles: {telemetry.ActiveStateHandles}");
    }

    [Fact]
    public async Task MultipleHandlerExecutions_ShouldAccumulateTelemetry()
    {
        // Arrange
        var actorId = "test-actor-multi";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);
        var request = new TestRequest { Operand = 1.0f };
        const int iterations = 10;

        var telemetryBefore = _bridge.GetTelemetry();

        // Act
        for (int i = 0; i < iterations; i++)
        {
            await _bridge.ExecuteHandlerAsync<TestRequest, TestResponse, TestState>(
                "TestKernel",
                handlerId: 1,
                request,
                handle);
        }

        var telemetryAfter = _bridge.GetTelemetry();

        // Assert
        Assert.Equal(
            telemetryBefore.TotalExecutions + iterations,
            telemetryAfter.TotalExecutions);
        Assert.Equal(
            telemetryBefore.CpuFallbackExecutions + iterations,
            telemetryAfter.CpuFallbackExecutions);

        _output.WriteLine($"✅ Telemetry accumulated after {iterations} executions");
        _output.WriteLine($"   Total executions: {telemetryAfter.TotalExecutions}");
        _output.WriteLine($"   Avg latency: {telemetryAfter.AverageLatencyNs:F2} ns");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task ConcurrentStateOperations_ShouldBeThreadSafe()
    {
        // Arrange
        const int concurrentOps = 20;
        var tasks = new List<Task>();

        // Act - Create multiple actors concurrently
        for (int i = 0; i < concurrentOps; i++)
        {
            var actorId = $"concurrent-actor-{i}";
            tasks.Add(Task.Run(async () =>
            {
                var handle = await _bridge.AllocateStateAsync<TestState>(actorId, -1);
                await _bridge.WriteStateAsync(handle, new TestState { Value = i, Counter = i * 10 });
                var state = await _bridge.ReadStateAsync(handle);
                await _bridge.ReleaseStateAsync(handle);
            }));
        }

        await Task.WhenAll(tasks);

        // Assert
        var telemetry = _bridge.GetTelemetry();
        Assert.Equal(0, telemetry.ActiveStateHandles); // All released

        _output.WriteLine($"✅ {concurrentOps} concurrent operations completed successfully");
        _output.WriteLine($"   Active handles after: {telemetry.ActiveStateHandles}");
    }

    public void Dispose()
    {
        _output.WriteLine("✅ CpuFallbackRingKernelBridge tests completed");
    }

    /// <summary>
    /// Test state structure for ring kernel bridge tests.
    /// </summary>
    private struct TestState
    {
        public float Value;
        public int Counter;
    }

    /// <summary>
    /// Test request structure.
    /// </summary>
    private struct TestRequest
    {
        public float Operand;
    }

    /// <summary>
    /// Test response structure.
    /// </summary>
    private struct TestResponse
    {
        public float Result;
        public int ErrorCode;
    }
}
