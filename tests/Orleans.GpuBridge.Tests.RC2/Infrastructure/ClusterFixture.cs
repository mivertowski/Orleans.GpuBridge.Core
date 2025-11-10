using System.Collections.Concurrent;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Grains.Batch;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.BackendProviders.Models;
using Orleans.Hosting;
using Orleans.TestingHost;

namespace Orleans.GpuBridge.Tests.RC2.Infrastructure;

/// <summary>
/// Orleans test cluster fixture for grain testing.
/// Provides a fully configured in-memory Orleans cluster with GPU Bridge components.
/// </summary>
public sealed class ClusterFixture : IDisposable
{
    private bool _disposed;

    /// <summary>
    /// Gets the Orleans test cluster instance.
    /// </summary>
    public TestCluster Cluster { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ClusterFixture"/> class.
    /// </summary>
    public ClusterFixture()
    {
        var builder = new TestClusterBuilder();

        // Configure silo
        builder.AddSiloBuilderConfigurator<TestSiloConfigurator>();

        // Configure client
        builder.AddClientBuilderConfigurator<TestClientConfigurator>();

        // Build and start cluster
        Cluster = builder.Build();
        Cluster.Deploy();
    }

    /// <summary>
    /// Silo configurator for test cluster.
    /// </summary>
    private sealed class TestSiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            // Configure in-memory grain storage
            siloBuilder.AddMemoryGrainStorage("gpuStore");
            siloBuilder.AddMemoryGrainStorage("PubSubStore");

            // Configure streams
            siloBuilder.AddMemoryStreams("Default");

            // Configure GPU Bridge with mock/test implementations
            siloBuilder.ConfigureServices(services =>
            {
                // Register GPU Bridge components
                services.AddSingleton<IGpuBridge, MockGpuBridge>();
                services.AddSingleton<ILogger<DeviceBroker>>(sp =>
                    sp.GetRequiredService<ILoggerFactory>().CreateLogger<DeviceBroker>());
                services.AddSingleton<IOptions<GpuBridgeOptions>>(Options.Create(new GpuBridgeOptions()));

                // Register and initialize DeviceBroker
                services.AddSingleton<DeviceBroker>(sp =>
                {
                    var logger = sp.GetRequiredService<ILogger<DeviceBroker>>();
                    var options = sp.GetRequiredService<IOptions<GpuBridgeOptions>>();
                    var broker = new DeviceBroker(logger, options);

                    // Initialize synchronously in test environment
                    broker.InitializeAsync(CancellationToken.None).GetAwaiter().GetResult();

                    return broker;
                });

                // Configure logging
                services.AddLogging(logging =>
                {
                    logging.SetMinimumLevel(LogLevel.Information);
                    logging.AddDebug();
                });
            });
        }
    }

    /// <summary>
    /// Client configurator for test cluster.
    /// </summary>
    private sealed class TestClientConfigurator : IClientBuilderConfigurator
    {
        public void Configure(IConfiguration configuration, IClientBuilder clientBuilder)
        {
            // Configure streams on client
            clientBuilder.AddMemoryStreams("Default");
        }
    }

    /// <summary>
    /// Disposes the cluster and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            Cluster?.StopAllSilos();
        }
        catch
        {
            // Ignore cleanup errors
        }

        _disposed = true;
    }
}

/// <summary>
/// Mock GPU Bridge implementation for testing.
/// Provides CPU-based fallback for GPU operations during tests.
/// Thread-safe for concurrent kernel access.
/// </summary>
internal sealed class MockGpuBridge : IGpuBridge
{
    private readonly ConcurrentDictionary<KernelId, object> _kernels = new();

    public ValueTask<IGpuKernel<TIn, TOut>> GetKernelAsync<TIn, TOut>(
        KernelId kernelId,
        CancellationToken cancellationToken = default)
        where TIn : notnull
        where TOut : notnull
    {
        // GetOrAdd is atomic and thread-safe
        var kernel = _kernels.GetOrAdd(kernelId, _ => new MockGpuKernel<TIn, TOut>(kernelId));
        return ValueTask.FromResult((IGpuKernel<TIn, TOut>)kernel);
    }

    public ValueTask<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken cancellationToken = default)
    {
        var devices = new List<GpuDevice>
        {
            new GpuDevice(
                Index: 0,
                Name: "Test GPU Device",
                Type: Orleans.GpuBridge.Abstractions.Enums.DeviceType.CPU,
                TotalMemoryBytes: 8L * 1024 * 1024 * 1024, // 8 GB
                AvailableMemoryBytes: 7L * 1024 * 1024 * 1024, // 7 GB available
                ComputeUnits: 8,
                Capabilities: Array.Empty<string>())
        };
        return ValueTask.FromResult<IReadOnlyList<GpuDevice>>(devices);
    }

    public ValueTask<GpuBridgeInfo> GetInfoAsync(CancellationToken ct = default)
    {
        var info = new GpuBridgeInfo(
            Version: "1.0.0-test",
            DeviceCount: 1,
            TotalMemoryBytes: 8L * 1024 * 1024 * 1024,
            Backend: Orleans.GpuBridge.Abstractions.Enums.GpuBackend.CPU,
            IsGpuAvailable: false,
            Metadata: null);

        return ValueTask.FromResult(info);
    }

    public ValueTask<object> ExecuteKernelAsync(string kernelId, object input, CancellationToken ct = default)
    {
        // Mock implementation - returns input as output for testing
        return ValueTask.FromResult(input);
    }
}

/// <summary>
/// Mock GPU Kernel implementation for testing.
/// Simulates GPU kernel execution using CPU operations.
/// Thread-safe for concurrent pipeline execution.
/// </summary>
internal sealed class MockGpuKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly KernelId _kernelId;
    private readonly ConcurrentDictionary<string, IReadOnlyList<TIn>> _batches = new();

    public MockGpuKernel(KernelId kernelId)
    {
        _kernelId = kernelId;
    }

    public async ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints = null,
        CancellationToken cancellationToken = default)
    {
        // Simulate async GPU work
        await Task.Delay(10, cancellationToken);

        var handle = KernelHandle.Create();
        _batches[handle.Id] = batch;

        return handle;
    }

    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Simulate reading results
        await Task.Delay(5, cancellationToken);

        // Retrieve the batch associated with this handle
        if (!_batches.TryGetValue(handle.Id, out var batch))
        {
            throw new ArgumentException($"Invalid kernel handle: {handle.Id}", nameof(handle));
        }

        try
        {
            // For testing, return one result per input item
            // This matches the expected 1:1 mapping for most GPU kernels
            for (int i = 0; i < batch.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Mock result creation based on output type
                // For float results, return mock values (simulate x * 2)
                if (typeof(TOut) == typeof(float))
                {
                    if (typeof(TIn) == typeof(float) && batch[i] is float inputValue)
                    {
                        // Simulate a simple transformation: input * 2
                        yield return (TOut)(object)(inputValue * 2.0f);
                    }
                    else
                    {
                        // If input type is not float, use index-based value
                        yield return (TOut)(object)(i * 2.0f);
                    }
                }
                else if (typeof(TOut) == typeof(int))
                {
                    if (typeof(TIn) == typeof(int) && batch[i] is int inputIntValue)
                    {
                        yield return (TOut)(object)(inputIntValue * 2);
                    }
                    else
                    {
                        yield return (TOut)(object)(i * 2);
                    }
                }
                else
                {
                    // For other types, try to create default instance
                    yield return default!;
                }
            }
        }
        finally
        {
            // Clean up the batch after reading results
            // Use TryRemove to handle thread-safe removal
            _batches.TryRemove(handle.Id, out _);
        }
    }

    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        var info = new KernelInfo(
            Id: _kernelId,
            Description: "Mock GPU kernel for testing",
            InputType: typeof(TIn),
            OutputType: typeof(TOut),
            SupportsGpu: false, // CPU fallback for testing
            PreferredBatchSize: 128,
            Metadata: null);

        return ValueTask.FromResult(info);
    }
}
