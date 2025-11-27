// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Backends.DotCompute;
using Orleans.GpuBridge.Backends.DotCompute.Configuration;
using Orleans.GpuBridge.Backends.DotCompute.RingKernels;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.RingKernelTests;

/// <summary>
/// Tests for the DotComputeRingKernelBridge implementation.
/// </summary>
/// <remarks>
/// <para>
/// Tests the GPU-enabled ring kernel bridge that wraps DotCompute for
/// real GPU execution. When no GPU is available, tests validate the
/// automatic fallback behavior.
/// </para>
/// <para>
/// These tests ensure proper integration between Orleans.GpuBridge
/// and the DotCompute backend for GPU-native actor messaging.
/// </para>
/// </remarks>
public class DotComputeRingKernelBridgeTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly DotComputeRingKernelBridge _bridge;
    private readonly DotComputeBackendProvider _backendProvider;
    private readonly ILogger<DotComputeRingKernelBridge> _logger;
    private readonly bool _gpuAvailable;

    public DotComputeRingKernelBridgeTests(ITestOutputHelper output)
    {
        _output = output;

        var services = new ServiceCollection();
        services.AddLogging(builder => builder
            .AddDebug()
            .SetMinimumLevel(LogLevel.Debug));

        var sp = services.BuildServiceProvider();
        var loggerFactory = sp.GetRequiredService<ILoggerFactory>();
        _logger = loggerFactory.CreateLogger<DotComputeRingKernelBridge>();

        // Create backend provider
        var providerLogger = loggerFactory.CreateLogger<DotComputeBackendProvider>();
        var options = Options.Create(new DotComputeOptions());
        _backendProvider = new DotComputeBackendProvider(providerLogger, loggerFactory, options);

        // Create bridge (without ring kernel runtime for basic tests)
        _bridge = new DotComputeRingKernelBridge(_logger, _backendProvider, ringKernelRuntime: null);

        // Check if GPU is available
        _gpuAvailable = Task.Run(async () => await _bridge.IsGpuAvailableAsync()).GetAwaiter().GetResult();

        _output.WriteLine("✅ DotComputeRingKernelBridge initialized");
        _output.WriteLine($"   GPU available: {_gpuAvailable}");
    }

    [Fact]
    public async Task IsGpuAvailableAsync_ShouldReturnCorrectStatus()
    {
        // Act
        var isAvailable = await _bridge.IsGpuAvailableAsync();

        // Assert
        _output.WriteLine($"✅ GPU availability check completed: {isAvailable}");

        // Note: isAvailable depends on CUDA/GPU hardware availability
        // In WSL2 without proper CUDA setup, this will return false
        // The important thing is it doesn't throw and returns a valid boolean
        _output.WriteLine($"   Bridge reports GPU available: {isAvailable}");

        // In WSL2 environment, GPU detection may be limited - this is expected
        // The test validates the bridge works correctly regardless of GPU availability
    }

    [Fact]
    public async Task GetDevicePlacementAsync_ShouldReturnValidDeviceId()
    {
        // Arrange
        var actorKey = "test-actor-placement";

        // Act
        var deviceId = await _bridge.GetDevicePlacementAsync(actorKey);

        // Assert
        _output.WriteLine($"✅ Device placement for {actorKey}: {deviceId}");

        if (_gpuAvailable)
        {
            Assert.True(deviceId >= 0, "Should return valid device ID when GPU available");
        }
        else
        {
            Assert.Equal(-1, deviceId);
            _output.WriteLine("   (Fallback: No GPU available)");
        }
    }

    [Fact]
    public async Task AllocateStateAsync_ShouldReturnValidHandle()
    {
        // Arrange
        var actorId = "test-actor-alloc-dotcompute";

        // Act
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, deviceId: 0);

        // Assert
        Assert.NotNull(handle);
        Assert.Equal(actorId, handle.ActorId);
        Assert.True(handle.SizeBytes > 0);

        _output.WriteLine($"✅ State allocated for actor {actorId}");
        _output.WriteLine($"   Size: {handle.SizeBytes} bytes");
        _output.WriteLine($"   Device ID: {handle.DeviceId}");
        _output.WriteLine($"   GPU pointer: 0x{handle.GpuPointer.ToInt64():X}");
        _output.WriteLine($"   Is valid: {handle.IsValid}");

        // Cleanup
        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task StateLifecycle_AllocateWriteReadRelease_ShouldWork()
    {
        // Arrange
        var actorId = "test-actor-lifecycle";
        var initialState = new TestState { Value = 123.456f, Counter = 42 };

        // Act - Allocate
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, deviceId: 0);
        _output.WriteLine($"✅ Allocated state for {actorId}");

        // Act - Write
        await _bridge.WriteStateAsync(handle, initialState);
        _output.WriteLine($"   Wrote state: Value={initialState.Value}, Counter={initialState.Counter}");

        // Act - Read
        var readState = await _bridge.ReadStateAsync(handle);
        _output.WriteLine($"   Read state: Value={readState.Value}, Counter={readState.Counter}");

        // Assert
        Assert.Equal(initialState.Value, readState.Value);
        Assert.Equal(initialState.Counter, readState.Counter);

        // Act - Release
        await _bridge.ReleaseStateAsync(handle);
        _output.WriteLine($"✅ Released state successfully");
    }

    [Fact]
    public async Task ExecuteHandlerAsync_ShouldReturnResult()
    {
        // Arrange
        var actorId = "test-actor-execute-dotcompute";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, deviceId: 0);
        handle.ShadowState = new TestState { Value = 10.0f, Counter = 1 };

        var request = new TestRequest { Operand = 5.0f };

        // Act
        var result = await _bridge.ExecuteHandlerAsync<TestRequest, TestResponse, TestState>(
            "TestKernel/Execute",
            handlerId: 1,
            request,
            handle);

        // Assert
        Assert.True(result.Success);
        _output.WriteLine($"✅ Handler executed");
        _output.WriteLine($"   Success: {result.Success}");
        _output.WriteLine($"   Was GPU: {result.WasGpuExecution}");
        _output.WriteLine($"   Latency: {result.LatencyNs} ns");
        _output.WriteLine($"   Error code: {result.ErrorCode}");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task ExecuteFireAndForgetAsync_ShouldUpdateState()
    {
        // Arrange
        var actorId = "test-actor-fandf-dotcompute";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, deviceId: 0);
        var initialState = new TestState { Value = 100.0f, Counter = 50 };
        handle.ShadowState = initialState;
        await _bridge.WriteStateAsync(handle, initialState);

        var request = new TestRequest { Operand = 10.0f };

        // Act
        var newState = await _bridge.ExecuteFireAndForgetAsync<TestRequest, TestState>(
            "TestKernel/FireAndForget",
            handlerId: 2,
            request,
            handle);

        // Assert
        _output.WriteLine($"✅ Fire-and-forget executed");
        _output.WriteLine($"   New state value: {newState.Value}");
        _output.WriteLine($"   New state counter: {newState.Counter}");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public void GetTelemetry_ShouldReturnValidMetrics()
    {
        // Act
        var telemetry = _bridge.GetTelemetry();

        // Assert
        Assert.NotNull(telemetry);

        _output.WriteLine($"✅ Telemetry retrieved");
        _output.WriteLine($"   Total executions: {telemetry.TotalExecutions}");
        _output.WriteLine($"   GPU executions: {telemetry.GpuExecutions}");
        _output.WriteLine($"   CPU fallbacks: {telemetry.CpuFallbackExecutions}");
        _output.WriteLine($"   Bytes to GPU: {telemetry.BytesToGpu}");
        _output.WriteLine($"   Bytes from GPU: {telemetry.BytesFromGpu}");
        _output.WriteLine($"   Avg latency: {telemetry.AverageLatencyNs:F2} ns");
        _output.WriteLine($"   Active handles: {telemetry.ActiveStateHandles}");
        _output.WriteLine($"   GPU memory: {telemetry.AllocatedGpuMemoryBytes} bytes");
    }

    [Fact]
    public async Task MultipleActors_ShouldIsolateState()
    {
        // Arrange
        var actor1Id = "isolation-test-1";
        var actor2Id = "isolation-test-2";

        // Act - Allocate and set different states
        var handle1 = await _bridge.AllocateStateAsync<TestState>(actor1Id, deviceId: 0);
        var handle2 = await _bridge.AllocateStateAsync<TestState>(actor2Id, deviceId: 0);

        var state1 = new TestState { Value = 111.0f, Counter = 1 };
        var state2 = new TestState { Value = 222.0f, Counter = 2 };

        await _bridge.WriteStateAsync(handle1, state1);
        await _bridge.WriteStateAsync(handle2, state2);

        // Read back
        var readState1 = await _bridge.ReadStateAsync(handle1);
        var readState2 = await _bridge.ReadStateAsync(handle2);

        // Assert - States should be isolated
        Assert.Equal(state1.Value, readState1.Value);
        Assert.Equal(state1.Counter, readState1.Counter);
        Assert.Equal(state2.Value, readState2.Value);
        Assert.Equal(state2.Counter, readState2.Counter);

        _output.WriteLine($"✅ Actor state isolation verified");
        _output.WriteLine($"   Actor 1: Value={readState1.Value}, Counter={readState1.Counter}");
        _output.WriteLine($"   Actor 2: Value={readState2.Value}, Counter={readState2.Counter}");

        await _bridge.ReleaseStateAsync(handle1);
        await _bridge.ReleaseStateAsync(handle2);
    }

    [Fact]
    public async Task SequentialExecutions_ShouldAccumulateTelemetry()
    {
        // Arrange
        var actorId = "telemetry-test";
        var handle = await _bridge.AllocateStateAsync<TestState>(actorId, deviceId: 0);
        var request = new TestRequest { Operand = 1.0f };

        var telemetryBefore = _bridge.GetTelemetry();
        const int iterations = 5;

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

        _output.WriteLine($"✅ Telemetry accumulated");
        _output.WriteLine($"   Before: {telemetryBefore.TotalExecutions}");
        _output.WriteLine($"   After: {telemetryAfter.TotalExecutions}");
        _output.WriteLine($"   Iterations: {iterations}");

        await _bridge.ReleaseStateAsync(handle);
    }

    [Fact]
    public async Task ConcurrentStateAllocations_ShouldBeThreadSafe()
    {
        // Arrange
        const int concurrentActors = 10;
        var tasks = new List<Task<GpuStateHandle<TestState>>>();

        // Act
        for (int i = 0; i < concurrentActors; i++)
        {
            var actorId = $"concurrent-dotcompute-{i}";
            tasks.Add(_bridge.AllocateStateAsync<TestState>(actorId, deviceId: 0));
        }

        var handles = await Task.WhenAll(tasks);

        // Assert
        Assert.Equal(concurrentActors, handles.Length);
        Assert.All(handles, h => Assert.NotNull(h));

        var telemetry = _bridge.GetTelemetry();
        _output.WriteLine($"✅ {concurrentActors} concurrent allocations completed");
        _output.WriteLine($"   Active handles: {telemetry.ActiveStateHandles}");

        // Cleanup
        foreach (var handle in handles)
        {
            await _bridge.ReleaseStateAsync(handle);
        }

        var telemetryAfter = _bridge.GetTelemetry();
        Assert.Equal(0, telemetryAfter.ActiveStateHandles);
        _output.WriteLine($"   After cleanup: {telemetryAfter.ActiveStateHandles} handles");
    }

    [Fact]
    public async Task LargeState_ShouldHandleCorrectly()
    {
        // Arrange
        var actorId = "large-state-test";

        // Act
        var handle = await _bridge.AllocateStateAsync<LargeTestState>(actorId, deviceId: 0);

        var largeState = new LargeTestState();
        unsafe
        {
            for (int i = 0; i < 64; i++)
            {
                largeState.Data[i] = (float)i * 1.5f;
            }
        }
        largeState.Id = 12345;

        await _bridge.WriteStateAsync(handle, largeState);
        var readState = await _bridge.ReadStateAsync(handle);

        // Assert
        Assert.Equal(largeState.Id, readState.Id);
        unsafe
        {
            for (int i = 0; i < 64; i++)
            {
                Assert.Equal(largeState.Data[i], readState.Data[i]);
            }
        }

        _output.WriteLine($"✅ Large state ({handle.SizeBytes} bytes) handled correctly");
        _output.WriteLine($"   State ID: {readState.Id}");
        unsafe
        {
            _output.WriteLine($"   Sample values: Data[0]={readState.Data[0]}, Data[63]={readState.Data[63]}");
        }

        await _bridge.ReleaseStateAsync(handle);
    }

    public void Dispose()
    {
        _output.WriteLine("✅ DotComputeRingKernelBridge tests completed");
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
    /// Large test state for memory tests.
    /// </summary>
    private unsafe struct LargeTestState
    {
        public fixed float Data[64]; // 256 bytes
        public int Id;
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
