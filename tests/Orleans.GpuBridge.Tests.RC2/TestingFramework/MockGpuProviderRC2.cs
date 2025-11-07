using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Tests.RC2.TestingFramework;

/// <summary>
/// Enhanced mock GPU provider for RC2 error handling tests
/// </summary>
public class MockGpuProviderRC2 : IDisposable
{
    private readonly ILogger<MockGpuProviderRC2> _logger;
    private readonly ConcurrentDictionary<string, MockKernelState> _kernelStates = new();
    private bool _disposed;

    // Test configuration properties
    public bool SimulateOutOfMemory { get; set; }
    public bool SimulateGpuTimeout { get; set; }
    public bool SimulateGpuCrash { get; set; }
    public bool SimulateFragmentation { get; set; }
    public long AvailableMemory { get; set; } = 8L * 1024 * 1024 * 1024; // 8GB
    public long UsedMemory { get; private set; }
    public int AllocationAttempts { get; private set; }
    public int FallbackCount { get; set; }
    public bool HasCpuFallback { get; set; } = true;
    public TimeSpan ExecutionTimeout { get; set; } = TimeSpan.FromSeconds(30);

    public MockGpuProviderRC2(ILogger<MockGpuProviderRC2>? logger = null)
    {
        _logger = logger ?? CreateDefaultLogger();
    }

    private static ILogger<MockGpuProviderRC2> CreateDefaultLogger()
    {
        using var factory = LoggerFactory.Create(builder => builder.AddConsole());
        return factory.CreateLogger<MockGpuProviderRC2>();
    }

    public async Task<TOut> ExecuteKernelAsync<TIn, TOut>(
        string kernelId,
        TIn input,
        CancellationToken cancellationToken = default)
        where TIn : notnull
        where TOut : notnull
    {
        ThrowIfDisposed();

        // Simulate GPU crash
        if (SimulateGpuCrash)
        {
            _logger.LogError("Simulating GPU crash for kernel {KernelId}", kernelId);
            throw new InvalidOperationException("GPU device crashed during execution");
        }

        // Simulate GPU timeout
        if (SimulateGpuTimeout)
        {
            _logger.LogWarning("Simulating GPU timeout for kernel {KernelId}", kernelId);
            await Task.Delay(ExecutionTimeout + TimeSpan.FromSeconds(5), cancellationToken);
            throw new TimeoutException($"Kernel execution timeout after {ExecutionTimeout.TotalSeconds}s");
        }

        // Simulate out of memory
        if (SimulateOutOfMemory)
        {
            AllocationAttempts++;
            _logger.LogError("Simulating GPU out of memory for kernel {KernelId}", kernelId);
            throw new OutOfMemoryException("GPU device out of memory");
        }

        // Normal execution
        _logger.LogDebug("Executing kernel {KernelId} on GPU", kernelId);
        await Task.Delay(10, cancellationToken); // Simulate work

        // Return mock result
        if (typeof(TOut) == typeof(float))
            return (TOut)(object)42.0f;
        if (typeof(TOut) == typeof(float[]))
            return (TOut)(object)new[] { 1.0f, 2.0f, 3.0f };

        return default(TOut)!;
    }

    public long TryAllocateMemory(long bytes)
    {
        ThrowIfDisposed();
        AllocationAttempts++;

        if (SimulateOutOfMemory)
        {
            _logger.LogError("Simulating out of memory: requested {RequestedBytes}, available {AvailableBytes}",
                bytes, AvailableMemory - UsedMemory);
            throw new OutOfMemoryException($"Cannot allocate {bytes} bytes");
        }

        if (SimulateFragmentation && bytes > 1024 * 1024) // > 1MB
        {
            _logger.LogWarning("Simulating memory fragmentation");
            throw new InvalidOperationException("Memory fragmentation detected");
        }

        var available = AvailableMemory - UsedMemory;
        if (bytes > available)
        {
            throw new OutOfMemoryException($"Insufficient memory: requested {bytes}, available {available}");
        }

        UsedMemory += bytes;
        _logger.LogDebug("Allocated {Bytes} bytes, total used: {UsedMemory}/{AvailableMemory}",
            bytes, UsedMemory, AvailableMemory);

        return bytes;
    }

    public void FreeMemory(long bytes)
    {
        ThrowIfDisposed();
        UsedMemory = Math.Max(0, UsedMemory - bytes);
        _logger.LogDebug("Freed {Bytes} bytes, total used: {UsedMemory}/{AvailableMemory}",
            bytes, UsedMemory, AvailableMemory);
    }

    public void DefragmentMemory()
    {
        ThrowIfDisposed();
        _logger.LogInformation("Defragmenting GPU memory");
        SimulateFragmentation = false;
    }

    public void Reset()
    {
        ThrowIfDisposed();
        SimulateOutOfMemory = false;
        SimulateGpuTimeout = false;
        SimulateGpuCrash = false;
        SimulateFragmentation = false;
        UsedMemory = 0;
        AllocationAttempts = 0;
        FallbackCount = 0;
        _kernelStates.Clear();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MockGpuProviderRC2));
    }

    public void Dispose()
    {
        if (_disposed) return;

        _kernelStates.Clear();
        _disposed = true;
        _logger.LogDebug("MockGpuProviderRC2 disposed");
    }
}

/// <summary>
/// Enhanced mock kernel for RC2 testing with error simulation
/// </summary>
public class MockKernelRC2<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    private readonly KernelInfo _info;
    private readonly MockGpuProviderRC2? _provider;
    private readonly Dictionary<string, IReadOnlyList<TIn>> _batches = new();
    private readonly Func<IReadOnlyList<TIn>, IAsyncEnumerable<TOut>>? _customExecution;

    public bool SimulateFailure { get; set; }
    public Exception? ExceptionToThrow { get; set; }
    public TimeSpan ExecutionDelay { get; set; } = TimeSpan.FromMilliseconds(10);

    public MockKernelRC2(
        KernelInfo info,
        MockGpuProviderRC2? provider = null,
        Func<IReadOnlyList<TIn>, IAsyncEnumerable<TOut>>? customExecution = null)
    {
        _info = info;
        _provider = provider;
        _customExecution = customExecution;
    }

    public async ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        if (SimulateFailure && ExceptionToThrow != null)
            throw ExceptionToThrow;

        var handle = KernelHandle.Create();
        _batches[handle.Id] = items;

        await Task.Delay(5, ct); // Simulate submission overhead
        return handle;
    }

    public async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!_batches.TryGetValue(handle.Id, out var items))
            throw new ArgumentException("Invalid handle", nameof(handle));

        try
        {
            if (SimulateFailure && ExceptionToThrow != null)
                throw ExceptionToThrow;

            await Task.Delay(ExecutionDelay, ct);

            if (_customExecution != null)
            {
                await foreach (var result in _customExecution(items).WithCancellation(ct))
                {
                    yield return result;
                }
            }
            else
            {
                // Default: return mock results
                foreach (var _ in items)
                {
                    ct.ThrowIfCancellationRequested();

                    if (typeof(TOut) == typeof(float))
                        yield return (TOut)(object)42.0f;
                    else if (typeof(TOut) == typeof(float[]))
                        yield return (TOut)(object)new[] { 1.0f, 2.0f, 3.0f };
                    else
                        yield return default(TOut)!;
                }
            }
        }
        finally
        {
            _batches.Remove(handle.Id);
        }
    }

    public ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        return new ValueTask<KernelInfo>(_info);
    }
}

/// <summary>
/// State tracking for mock kernels
/// </summary>
internal class MockKernelState
{
    public int ExecutionCount { get; set; }
    public int FailureCount { get; set; }
    public DateTimeOffset LastExecution { get; set; }
    public TimeSpan TotalExecutionTime { get; set; }
}
