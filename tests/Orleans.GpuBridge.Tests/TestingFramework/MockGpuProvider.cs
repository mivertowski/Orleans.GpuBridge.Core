using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Runtime;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Mock GPU provider for testing without actual GPU hardware
/// </summary>
public class MockGpuProvider : IDisposable
{
    private readonly ILogger<MockGpuProvider> _logger;
    private readonly MockGpuConfiguration _config;
    private readonly ConcurrentDictionary<int, MockGpuDevice> _devices = new();
    private readonly ConcurrentDictionary<string, MockGpuContext> _contexts = new();
    private bool _disposed;

    public MockGpuProvider(ILogger<MockGpuProvider> logger, MockGpuConfiguration? config = null)
    {
        _logger = logger;
        _config = config ?? new MockGpuConfiguration();
        InitializeDevices();
    }

    public int DeviceCount => _devices.Count;
    public bool IsInitialized { get; private set; }

    public Task InitializeAsync()
    {
        if (IsInitialized) return Task.CompletedTask;

        _logger.LogInformation("Initializing mock GPU provider with {DeviceCount} devices", _config.DeviceCount);
        IsInitialized = true;
        return Task.CompletedTask;
    }

    public MockGpuDevice GetDevice(int deviceIndex)
    {
        ThrowIfDisposed();
        return _devices.TryGetValue(deviceIndex, out var device) ? device : throw new ArgumentOutOfRangeException(nameof(deviceIndex));
    }

    public IEnumerable<MockGpuDevice> GetDevices()
    {
        ThrowIfDisposed();
        return _devices.Values.OrderBy(d => d.Index);
    }

    public MockGpuContext CreateContext(int deviceIndex = 0)
    {
        ThrowIfDisposed();
        
        var device = GetDevice(deviceIndex);
        var context = new MockGpuContext(device, _logger, _config);
        var contextId = Guid.NewGuid().ToString();
        _contexts[contextId] = context;
        
        _logger.LogDebug("Created context {ContextId} for device {DeviceIndex}", contextId, deviceIndex);
        return context;
    }

    public void DestroyContext(string contextId)
    {
        if (_contexts.TryRemove(contextId, out var context))
        {
            context.Dispose();
            _logger.LogDebug("Destroyed context {ContextId}", contextId);
        }
    }

    private void InitializeDevices()
    {
        // Create CPU device (always present)
        _devices[0] = new MockGpuDevice(
            0, 
            "Mock CPU Device", 
            DeviceType.Cpu,
            _config.CpuMemoryBytes,
            Environment.ProcessorCount,
            new[] { "CPU", "AVX", "SSE" });

        // Create GPU devices
        for (int i = 1; i <= _config.DeviceCount - 1; i++)
        {
            _devices[i] = new MockGpuDevice(
                i,
                $"Mock GPU Device {i}",
                DeviceType.Gpu,
                _config.GpuMemoryBytes,
                _config.ComputeUnits,
                new[] { "CUDA", "OpenCL", "Vulkan" });
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MockGpuProvider));
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var context in _contexts.Values)
        {
            context.Dispose();
        }
        _contexts.Clear();
        _devices.Clear();

        _disposed = true;
        _logger.LogDebug("Mock GPU provider disposed");
    }
}

/// <summary>
/// Mock GPU device for testing
/// </summary>
public class MockGpuDevice
{
    public int Index { get; }
    public string Name { get; }
    public DeviceType Type { get; }
    public long TotalMemoryBytes { get; }
    public int ComputeUnits { get; }
    public string[] Capabilities { get; }
    
    public long UsedMemoryBytes { get; private set; }
    public double UtilizationPercentage { get; private set; }
    
    private readonly Random _random = new();

    public MockGpuDevice(int index, string name, DeviceType type, long totalMemoryBytes, int computeUnits, string[] capabilities)
    {
        Index = index;
        Name = name;
        Type = type;
        TotalMemoryBytes = totalMemoryBytes;
        ComputeUnits = computeUnits;
        Capabilities = capabilities;
    }

    public void AllocateMemory(long bytes)
    {
        if (UsedMemoryBytes + bytes > TotalMemoryBytes)
            throw new OutOfMemoryException($"Insufficient memory on device {Index}");
        
        UsedMemoryBytes += bytes;
    }

    public void FreeMemory(long bytes)
    {
        UsedMemoryBytes = Math.Max(0, UsedMemoryBytes - bytes);
    }

    public void SimulateUtilization()
    {
        // Simulate changing utilization
        UtilizationPercentage = Math.Max(0, Math.Min(100, UtilizationPercentage + _random.Next(-10, 10)));
    }

    public IComputeDevice ToDeviceInfo()
    {
        return new TestComputeDevice
        {
            DeviceId = $"mock-device-{Index}",
            Index = Index,
            Name = Name,
            Type = Type,
            Vendor = "Mock Vendor",
            Architecture = "Mock Architecture",
            ComputeCapability = new Version(7, 5),
            TotalMemoryBytes = TotalMemoryBytes,
            AvailableMemoryBytes = TotalMemoryBytes - UsedMemoryBytes,
            ComputeUnits = ComputeUnits,
            MaxClockFrequencyMHz = 1500,
            MaxThreadsPerBlock = 1024,
            MaxWorkGroupDimensions = new[] { 1024, 1024, 64 },
            WarpSize = 32,
            Properties = new Dictionary<string, object> { ["Capabilities"] = Capabilities }
        };
    }
}

/// <summary>
/// Mock GPU context for testing kernel execution
/// </summary>
public class MockGpuContext : IDisposable
{
    private readonly MockGpuDevice _device;
    private readonly ILogger _logger;
    private readonly MockGpuConfiguration _config;
    private readonly ConcurrentDictionary<string, MockGpuBuffer> _buffers = new();
    private readonly ConcurrentDictionary<string, MockGpuKernel> _kernels = new();
    private bool _disposed;

    public MockGpuContext(MockGpuDevice device, ILogger logger, MockGpuConfiguration config)
    {
        _device = device;
        _logger = logger;
        _config = config;
        Id = Guid.NewGuid().ToString();
    }

    public string Id { get; }
    public MockGpuDevice Device => _device;

    public MockGpuBuffer CreateBuffer<T>(int count) where T : struct
    {
        ThrowIfDisposed();
        
        var sizeBytes = count * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        _device.AllocateMemory(sizeBytes);
        
        var buffer = new MockGpuBuffer(Guid.NewGuid().ToString(), sizeBytes, _config);
        _buffers[buffer.Id] = buffer;
        
        _logger.LogDebug("Created buffer {BufferId} of size {SizeBytes} bytes", buffer.Id, sizeBytes);
        return buffer;
    }

    public MockGpuKernel CompileKernel(string name, string source)
    {
        ThrowIfDisposed();
        
        // Simulate compilation delay
        if (_config.KernelCompilationDelay > TimeSpan.Zero)
            Thread.Sleep(_config.KernelCompilationDelay);
        
        var kernel = new MockGpuKernel(Guid.NewGuid().ToString(), name, source, _config);
        _kernels[kernel.Id] = kernel;
        
        _logger.LogDebug("Compiled kernel {KernelId} with name {Name}", kernel.Id, name);
        return kernel;
    }

    public async Task ExecuteKernelAsync(MockGpuKernel kernel, int workGroupSize = 1)
    {
        ThrowIfDisposed();
        
        _device.SimulateUtilization();
        
        // Simulate execution time
        if (_config.KernelExecutionDelay > TimeSpan.Zero)
            await Task.Delay(_config.KernelExecutionDelay);
        
        // Simulate execution failure
        if (_config.ExecutionFailureRate > 0 && Random.Shared.NextDouble() < _config.ExecutionFailureRate)
            throw new InvalidOperationException($"Simulated execution failure for kernel {kernel.Name}");
        
        _logger.LogDebug("Executed kernel {KernelId} with work group size {WorkGroupSize}", kernel.Id, workGroupSize);
    }

    public void Synchronize()
    {
        ThrowIfDisposed();
        
        // Simulate synchronization delay
        if (_config.SynchronizationDelay > TimeSpan.Zero)
            Thread.Sleep(_config.SynchronizationDelay);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MockGpuContext));
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var buffer in _buffers.Values)
        {
            _device.FreeMemory(buffer.SizeBytes);
            buffer.Dispose();
        }
        _buffers.Clear();

        foreach (var kernel in _kernels.Values)
        {
            kernel.Dispose();
        }
        _kernels.Clear();

        _disposed = true;
        _logger.LogDebug("Mock GPU context {ContextId} disposed", Id);
    }
}

/// <summary>
/// Mock GPU buffer for testing memory operations
/// </summary>
public class MockGpuBuffer : IDisposable
{
    private readonly MockGpuConfiguration _config;
    private bool _disposed;

    public MockGpuBuffer(string id, long sizeBytes, MockGpuConfiguration config)
    {
        Id = id;
        SizeBytes = sizeBytes;
        _config = config;
    }

    public string Id { get; }
    public long SizeBytes { get; }

    public void Write<T>(T[] data) where T : struct
    {
        ThrowIfDisposed();
        
        // Simulate write delay
        if (_config.MemoryTransferDelay > TimeSpan.Zero)
            Thread.Sleep(_config.MemoryTransferDelay);
    }

    public void Read<T>(T[] data) where T : struct
    {
        ThrowIfDisposed();
        
        // Simulate read delay
        if (_config.MemoryTransferDelay > TimeSpan.Zero)
            Thread.Sleep(_config.MemoryTransferDelay);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MockGpuBuffer));
    }

    public void Dispose()
    {
        _disposed = true;
    }
}

/// <summary>
/// Mock GPU kernel for testing compute operations
/// </summary>
public class MockGpuKernel : IDisposable
{
    private readonly MockGpuConfiguration _config;
    private readonly List<object> _arguments = new();
    private bool _disposed;

    public MockGpuKernel(string id, string name, string source, MockGpuConfiguration config)
    {
        Id = id;
        Name = name;
        Source = source;
        _config = config;
    }

    public string Id { get; }
    public string Name { get; }
    public string Source { get; }

    public void SetArgument(int index, object argument)
    {
        ThrowIfDisposed();
        
        // Ensure list is large enough
        while (_arguments.Count <= index)
            _arguments.Add(null!);
        
        _arguments[index] = argument;
    }

    public T GetArgument<T>(int index)
    {
        ThrowIfDisposed();
        
        if (index >= _arguments.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
        
        return (T)_arguments[index];
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(MockGpuKernel));
    }

    public void Dispose()
    {
        _disposed = true;
    }
}

/// <summary>
/// Configuration for mock GPU provider behavior
/// </summary>
public class MockGpuConfiguration
{
    public int DeviceCount { get; set; } = 3; // 1 CPU + 2 GPU
    public long CpuMemoryBytes { get; set; } = 16L * 1024 * 1024 * 1024; // 16GB
    public long GpuMemoryBytes { get; set; } = 8L * 1024 * 1024 * 1024; // 8GB
    public int ComputeUnits { get; set; } = 32;
    
    public TimeSpan KernelCompilationDelay { get; set; } = TimeSpan.FromMilliseconds(10);
    public TimeSpan KernelExecutionDelay { get; set; } = TimeSpan.FromMilliseconds(5);
    public TimeSpan MemoryTransferDelay { get; set; } = TimeSpan.FromMilliseconds(1);
    public TimeSpan SynchronizationDelay { get; set; } = TimeSpan.FromMilliseconds(1);
    
    public double ExecutionFailureRate { get; set; } = 0.0; // 0% failure rate by default
}