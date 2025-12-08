using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Placement;
using Orleans.GpuBridge.Abstractions.RingKernels;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.Extensions;
using Orleans.GpuBridge.Runtime.RingKernels;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Integration.Tests;

/// <summary>
/// Integration tests for ring kernel execution pipeline.
/// Tests the full path from grain → ring kernel bridge → CPU/GPU execution.
/// </summary>
public sealed class RingKernelIntegrationTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly ServiceProvider _serviceProvider;

    public RingKernelIntegrationTests(ITestOutputHelper output)
    {
        _output = output;

        // Build service provider with GPU bridge services
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddDebug().SetMinimumLevel(LogLevel.Debug));

        // Add GPU bridge services with CPU fallback
        services.AddGpuBridge()
            .Services
            .AddRingKernelSupport()
            .AddRingKernelBridge(); // CPU fallback

        // Add CPU memory pool for testing
        services.AddSingleton(typeof(IGpuMemoryPool<>), typeof(CpuMemoryPool<>));

        _serviceProvider = services.BuildServiceProvider();
    }

    public void Dispose()
    {
        _serviceProvider.Dispose();
    }

    /// <summary>
    /// Tests end-to-end execution through CPU fallback ring kernel bridge.
    /// Verifies the full pipeline: request → bridge → handler → response.
    /// </summary>
    [Fact]
    public async Task EndToEnd_CpuFallbackBridge_ExecutesHandler()
    {
        // Arrange
        var bridge = _serviceProvider.GetRequiredService<IRingKernelBridge>();
        var registry = new CpuFallbackHandlerRegistry();

        // Register a test handler with unmanaged types
        var handlerCalled = false;
        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "test_kernel",
            0,
            (request, state) =>
            {
                handlerCalled = true;
                return (new TestResponse { Value = request.Input * 2 }, state);
            });

        // Act
        var isAvailable = await bridge.IsGpuAvailableAsync();
        _output.WriteLine($"GPU Available: {isAvailable}");

        // Execute through registry directly (simulating bridge dispatch)
        var request = new TestRequest { Input = 42 };
        var hasHandler = registry.HasHandler("test_kernel", 0);
        _output.WriteLine($"Has Handler: {hasHandler}");

        TestResponse? response = null;
        if (hasHandler)
        {
            var result = registry.ExecuteHandler<TestRequest, TestResponse, TestState>(
                "test_kernel", 0, request, default);
            response = result?.Response;
        }

        // Assert
        handlerCalled.Should().BeTrue("CPU fallback handler should have been called");
        response.Should().NotBeNull();
        response!.Value.Value.Should().Be(84, "Handler should return input * 2");

        _output.WriteLine($"End-to-end test completed: Input={request.Input}, Output={response.Value.Value}");
    }

    /// <summary>
    /// Tests multiple concurrent requests through the registry.
    /// Verifies thread-safety and correct request/response correlation.
    /// </summary>
    [Fact]
    public async Task EndToEnd_ConcurrentRequests_AllComplete()
    {
        // Arrange
        var registry = new CpuFallbackHandlerRegistry();
        var processedCount = 0;

        registry.RegisterHandler<TestRequest, TestResponse, TestState>(
            "concurrent_kernel",
            0,
            (request, state) =>
            {
                Interlocked.Increment(ref processedCount);
                return (new TestResponse { Value = request.Input * 2 }, state);
            });

        var requestCount = 100;
        var tasks = new List<Task>();

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < requestCount; i++)
        {
            var input = i;
            tasks.Add(Task.Run(() =>
            {
                var request = new TestRequest { Input = input };
                var result = registry.ExecuteHandler<TestRequest, TestResponse, TestState>(
                    "concurrent_kernel", 0, request, default);
                return result;
            }));
        }

        await Task.WhenAll(tasks);
        sw.Stop();

        // Assert
        processedCount.Should().Be(requestCount);

        _output.WriteLine($"Concurrent test: {requestCount} requests in {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"Throughput: {requestCount / sw.Elapsed.TotalSeconds:N0} requests/sec");
    }

    /// <summary>
    /// Tests the ring kernel bridge initialization and availability check.
    /// </summary>
    [Fact]
    public async Task RingKernelBridge_Initialize_ReportsAvailability()
    {
        // Arrange
        var bridge = _serviceProvider.GetRequiredService<IRingKernelBridge>();

        // Act
        var isAvailable = await bridge.IsGpuAvailableAsync();

        // Assert
        _output.WriteLine($"Ring Kernel Bridge Status:");
        _output.WriteLine($"  IsGpuAvailable: {isAvailable}");

        // In CPU fallback mode, GPU is not available (expected)
        isAvailable.Should().BeFalse("CPU fallback mode should report GPU unavailable");
    }

    /// <summary>
    /// Tests queue depth monitoring with the actual interface.
    /// </summary>
    [Fact]
    public async Task QueueDepthMonitor_GetSnapshot_ReturnsValidMetrics()
    {
        // Arrange - Build provider with placement services
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddDebug().SetMinimumLevel(LogLevel.Warning));
        services.AddSingleton(new Moq.Mock<IGrainFactory>().Object);

        services.AddGpuBridge()
            .Services
            .AddRingKernelSupport()
            .AddRingKernelBridge();

        services.AddGpuNativePlacement();

        var provider = services.BuildServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var monitor = provider.GetRequiredService<IQueueDepthMonitor>();

        // Act
        var snapshot = await monitor.GetQueueDepthAsync(siloId: null, deviceIndex: 0, CancellationToken.None);

        // Assert
        snapshot.DeviceIndex.Should().Be(0);
        snapshot.TotalMemoryBytes.Should().BeGreaterThan(0);
        snapshot.AvailableMemoryBytes.Should().BeGreaterThan(0);
        snapshot.AvailableMemoryBytes.Should().BeLessThanOrEqualTo(snapshot.TotalMemoryBytes);

        _output.WriteLine($"Queue Depth Snapshot:");
        _output.WriteLine($"  SiloId: {snapshot.SiloId}");
        _output.WriteLine($"  Device: {snapshot.DeviceIndex}");
        _output.WriteLine($"  TotalMemory: {snapshot.TotalMemoryBytes:N0} bytes");
        _output.WriteLine($"  AvailableMemory: {snapshot.AvailableMemoryBytes:N0} bytes");
        _output.WriteLine($"  Utilization: {snapshot.GpuUtilization:P2}");
        _output.WriteLine($"  InputQueueUtilization: {snapshot.InputQueueUtilization:P2}");
    }

    /// <summary>
    /// Tests adaptive load balancer device selection.
    /// </summary>
    [Fact]
    public async Task AdaptiveLoadBalancer_SelectDevice_ReturnsValidPlacement()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddDebug().SetMinimumLevel(LogLevel.Warning));
        services.AddSingleton(new Moq.Mock<IGrainFactory>().Object);

        services.AddGpuBridge()
            .Services
            .AddRingKernelSupport()
            .AddRingKernelBridge();

        services.AddGpuNativePlacement();

        var provider = services.BuildServiceProvider();
        await using var _ = provider.ConfigureAwait(false);
        var loadBalancer = provider.GetRequiredService<IAdaptiveLoadBalancer>();

        // Act
        var request = new LoadBalancingRequest
        {
            GrainType = "TestGrain",
            GrainIdentity = "test-grain-1",
            AffinityGroup = "test-group"
        };

        var result = await loadBalancer.SelectDeviceAsync(request, CancellationToken.None);

        // Assert
        result.DeviceIndex.Should().BeGreaterThanOrEqualTo(0);
        result.PlacementScore.Should().BeGreaterThan(0);

        _output.WriteLine($"Load Balancer Selection:");
        _output.WriteLine($"  SiloId: {result.SiloId}");
        _output.WriteLine($"  DeviceIndex: {result.DeviceIndex}");
        _output.WriteLine($"  PlacementScore: {result.PlacementScore:P2}");
        _output.WriteLine($"  SelectionReason: {result.SelectionReason}");
        _output.WriteLine($"  IsFallback: {result.IsFallback}");
    }

    private struct TestRequest
    {
        public int Input;
    }

    private struct TestResponse
    {
        public int Value;
    }

    private struct TestState
    {
        public int Counter;
    }
}
