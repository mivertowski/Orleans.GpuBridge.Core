using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime.Providers;
using Xunit;

namespace Orleans.GpuBridge.Tests.Providers;

public class GpuBackendRegistryTests
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IGpuBackendRegistry _registry;

    public GpuBackendRegistryTests()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        
        // Add test providers
        services.AddSingleton<IGpuBackendProvider, TestGpuProvider>();
        services.AddSingleton<IGpuBackendProvider, TestCpuProvider>();
        
        _serviceProvider = services.BuildServiceProvider();
        _registry = new GpuBackendRegistry(_serviceProvider, _serviceProvider.GetRequiredService<ILogger<GpuBackendRegistry>>());
    }

    [Fact]
    public async Task InitializeAsync_RegistersAllAvailableProviders()
    {
        // Act
        await _registry.InitializeAsync();
        
        // Assert
        var providers = await _registry.GetRegisteredProvidersAsync();
        Assert.Equal(2, providers.Count());
        Assert.Contains(providers, p => p.ProviderId == "TestGpu");
        Assert.Contains(providers, p => p.ProviderId == "TestCpu");
    }

    [Fact]
    public async Task SelectProviderAsync_ReturnsProviderMatchingCriteria()
    {
        // Arrange
        await _registry.InitializeAsync();
        var criteria = new ProviderSelectionCriteria
        {
            PreferredBackends = new[] { GpuBackend.Cuda },
            RequiredCapabilities = new BackendCapabilities
            {
                SupportedBackends = new[] { GpuBackend.Cuda }
            }
        };

        // Act
        var provider = await _registry.SelectProviderAsync(criteria);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestGpu", provider.ProviderId);
    }

    [Fact]
    public async Task SelectProviderAsync_ReturnsCpuFallbackWhenNoGpuAvailable()
    {
        // Arrange
        await _registry.InitializeAsync();
        var criteria = new ProviderSelectionCriteria
        {
            PreferredBackends = new[] { GpuBackend.DirectCompute }, // Not supported by test providers
            AllowCpuFallback = true
        };

        // Act
        var provider = await _registry.SelectProviderAsync(criteria);

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestCpu", provider.ProviderId);
    }

    [Fact]
    public async Task SelectProviderAsync_ReturnsNullWhenNoCpuFallback()
    {
        // Arrange
        await _registry.InitializeAsync();
        var criteria = new ProviderSelectionCriteria
        {
            PreferredBackends = new[] { GpuBackend.DirectCompute },
            AllowCpuFallback = false
        };

        // Act
        var provider = await _registry.SelectProviderAsync(criteria);

        // Assert
        Assert.Null(provider);
    }

    [Fact]
    public async Task GetProviderByIdAsync_ReturnsCorrectProvider()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var provider = await _registry.GetProviderByIdAsync("TestGpu");

        // Assert
        Assert.NotNull(provider);
        Assert.Equal("TestGpu", provider.ProviderId);
    }

    [Fact]
    public async Task IsProviderAvailableAsync_ReturnsTrueForRegisteredProvider()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var isAvailable = await _registry.IsProviderAvailableAsync("TestGpu");

        // Assert
        Assert.True(isAvailable);
    }

    [Fact]
    public async Task IsProviderAvailableAsync_ReturnsFalseForUnregisteredProvider()
    {
        // Arrange
        await _registry.InitializeAsync();

        // Act
        var isAvailable = await _registry.IsProviderAvailableAsync("NonExistent");

        // Assert
        Assert.False(isAvailable);
    }
}

// Test provider implementations
internal class TestGpuProvider : IGpuBackendProvider
{
    public string ProviderId => "TestGpu";
    public BackendCapabilities Capabilities => new()
    {
        SupportedBackends = new[] { GpuBackend.Cuda },
        SupportedDataTypes = new[] { typeof(float), typeof(int) },
        SupportsJitCompilation = true
    };

    public bool IsAvailable() => true;

    public Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public IDeviceManager GetDeviceManager() => new TestDeviceManager();
    public IKernelCompiler GetKernelCompiler() => new TestKernelCompiler();
    public IMemoryAllocator GetMemoryAllocator() => new TestMemoryAllocator();
    public IKernelExecutor GetKernelExecutor() => new TestKernelExecutor();
    public ICommandQueue GetDefaultCommandQueue() => new TestCommandQueue();

    public Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new HealthCheckResult(true, "Healthy"));
    }

    public void Dispose() { }
}

internal class TestCpuProvider : IGpuBackendProvider
{
    public string ProviderId => "TestCpu";
    public BackendCapabilities Capabilities => BackendCapabilities.CreateCpuFallback();

    public bool IsAvailable() => true;

    public Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    public IDeviceManager GetDeviceManager() => new TestDeviceManager();
    public IKernelCompiler GetKernelCompiler() => new TestKernelCompiler();
    public IMemoryAllocator GetMemoryAllocator() => new TestMemoryAllocator();
    public IKernelExecutor GetKernelExecutor() => new TestKernelExecutor();
    public ICommandQueue GetDefaultCommandQueue() => new TestCommandQueue();

    public Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new HealthCheckResult(true, "Healthy"));
    }

    public void Dispose() { }
}

// Test stub implementations
internal class TestDeviceManager : IDeviceManager
{
    public Task InitializeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IEnumerable<IComputeDevice> GetDevices() => new[] { new TestComputeDevice() };
    public IComputeDevice GetDefaultDevice() => new TestComputeDevice();
    public IComputeDevice? GetDevice(string deviceId) => new TestComputeDevice();
    public IEnumerable<IComputeDevice> GetDevicesByType(DeviceType deviceType) => new[] { new TestComputeDevice() };
    public Task<DeviceHealthInfo> GetDeviceHealthAsync(string deviceId, CancellationToken cancellationToken = default) =>
        Task.FromResult(new DeviceHealthInfo(true, 50.0, 65.0, null));
    public void Dispose() { }
}

internal class TestComputeDevice : IComputeDevice
{
    public string Id => "test-device";
    public string Name => "Test Device";
    public DeviceType Type => DeviceType.Gpu;
    public int ComputeUnits => 16;
    public int MaxWorkGroupSize => 256;
    public long MaxMemoryBytes => 1024 * 1024 * 1024;
    public bool IsHealthy => true;
    public string? LastError => null;
    public void Dispose() { }
}

internal class TestKernelCompiler : IKernelCompiler
{
    public Task<CompiledKernel> CompileAsync(KernelSource source, KernelCompilationOptions options, CancellationToken cancellationToken = default) =>
        Task.FromResult(new TestCompiledKernel(source.Name));
    public Task<CompiledKernel> CompileFromFileAsync(string filePath, string kernelName, KernelCompilationOptions options, CancellationToken cancellationToken = default) =>
        Task.FromResult(new TestCompiledKernel(kernelName));
    public bool IsKernelCached(string kernelId) => false;
    public CompiledKernel? GetCachedKernel(string kernelId) => null;
    public void ClearCache() { }
    public void Dispose() { }
}

internal class TestCompiledKernel : CompiledKernel
{
    public TestCompiledKernel(string name) : base($"test-{name}", name, new TestComputeDevice(), new Dictionary<string, object>()) { }
}

internal class TestMemoryAllocator : IMemoryAllocator
{
    public Task<IDeviceMemory> AllocateAsync(long sizeBytes, MemoryAllocationOptions options, CancellationToken cancellationToken = default) =>
        Task.FromResult<IDeviceMemory>(new TestDeviceMemory(sizeBytes));
    public Task<IDeviceMemory<T>> AllocateAsync<T>(int elementCount, MemoryAllocationOptions options, CancellationToken cancellationToken = default) where T : unmanaged =>
        Task.FromResult<IDeviceMemory<T>>(new TestDeviceMemory<T>(elementCount));
    public Task<IPinnedMemory> AllocatePinnedAsync(long sizeBytes, CancellationToken cancellationToken = default) =>
        Task.FromResult<IPinnedMemory>(new TestPinnedMemory(sizeBytes));
    public Task<IUnifiedMemory> AllocateUnifiedAsync(long sizeBytes, UnifiedMemoryOptions options, CancellationToken cancellationToken = default) =>
        Task.FromResult<IUnifiedMemory>(new TestUnifiedMemory(sizeBytes));
    public MemoryPoolStatistics GetPoolStatistics() => new(0, 0, 0, 0, 0, 0, 0.0);
    public Task CompactAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task ResetAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public void Dispose() { }
}

internal class TestDeviceMemory : IDeviceMemory
{
    public TestDeviceMemory(long sizeBytes) { SizeBytes = sizeBytes; }
    public IntPtr DevicePointer => IntPtr.Zero;
    public IComputeDevice Device => new TestComputeDevice();
    public long SizeBytes { get; }
    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) => new TestDeviceMemory(sizeBytes);
    public void Dispose() { }
}

internal class TestDeviceMemory<T> : TestDeviceMemory, IDeviceMemory<T> where T : unmanaged
{
    public TestDeviceMemory(int elementCount) : base(elementCount * System.Runtime.CompilerServices.Unsafe.SizeOf<T>()) 
    { 
        ElementCount = elementCount; 
    }
    public int ElementCount { get; }
    public Task CopyFromHostAsync(ReadOnlySpan<T> hostData, int destinationOffset = 0, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(Span<T> hostData, int sourceOffset = 0, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public new IDeviceMemory<T> CreateView(int offsetElements, int elementCount) => new TestDeviceMemory<T>(elementCount);
}

internal class TestPinnedMemory : IPinnedMemory
{
    public TestPinnedMemory(long sizeBytes) { SizeBytes = sizeBytes; }
    public long SizeBytes { get; }
    public IntPtr HostPointer => IntPtr.Zero;
    public Span<byte> AsSpan() => new byte[SizeBytes];
    public Task RegisterWithDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task UnregisterFromDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public void Dispose() { }
}

internal class TestUnifiedMemory : TestDeviceMemory, IUnifiedMemory
{
    public TestUnifiedMemory(long sizeBytes) : base(sizeBytes) { }
    public IntPtr HostPointer => IntPtr.Zero;
    public Task PrefetchAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task AdviseAsync(MemoryAdvice advice, IComputeDevice? device = null, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Span<byte> AsHostSpan() => new byte[SizeBytes];
}

internal class TestKernelExecutor : IKernelExecutor
{
    public Task<KernelExecutionResult> ExecuteAsync(CompiledKernel kernel, KernelExecutionParameters parameters, CancellationToken cancellationToken = default) =>
        Task.FromResult(new KernelExecutionResult(true));
    public Task<IKernelExecution> ExecuteAsyncNonBlocking(CompiledKernel kernel, KernelExecutionParameters parameters, CancellationToken cancellationToken = default) =>
        Task.FromResult<IKernelExecution>(new TestKernelExecution());
    public Task<BatchExecutionResult> ExecuteBatchAsync(IReadOnlyList<KernelBatchItem> batch, BatchExecutionOptions options, CancellationToken cancellationToken = default) =>
        Task.FromResult(new BatchExecutionResult(batch.Count, 0, new List<KernelExecutionResult>(), TimeSpan.Zero));
    public IKernelGraph CreateGraph(string graphName) => new TestKernelGraph();
    public Task<KernelProfile> ProfileAsync(CompiledKernel kernel, KernelExecutionParameters parameters, int iterations = 100, CancellationToken cancellationToken = default) =>
        Task.FromResult(new KernelProfile(TimeSpan.FromMilliseconds(1), TimeSpan.FromMilliseconds(1), TimeSpan.FromMilliseconds(1), 0.1, 0, 0, 256));
    public ExecutionStatistics GetStatistics() => new(0, 0, 0, TimeSpan.Zero, TimeSpan.Zero, 0, 0, new Dictionary<string, long>());
    public void ResetStatistics() { }
    public void Dispose() { }
}

internal class TestKernelExecution : IKernelExecution
{
    public string ExecutionId => "test-execution";
    public bool IsCompleted => true;
    public bool IsCanceled => false;
    public Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(new KernelExecutionResult(true));
    public void Cancel() { }
    public void Dispose() { }
}

internal class TestKernelGraph : IKernelGraph
{
    public string GraphName => "test-graph";
    public IReadOnlyList<KernelGraphNode> Nodes => new List<KernelGraphNode>();
    public IKernelGraph AddNode(string nodeId, CompiledKernel kernel, KernelExecutionParameters parameters) => this;
    public IKernelGraph AddDependency(string fromNodeId, string toNodeId) => this;
    public Task<GraphExecutionResult> ExecuteAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(new GraphExecutionResult("test-graph", true, new Dictionary<string, KernelExecutionResult>(), TimeSpan.Zero));
    public void Dispose() { }
}

internal class TestCommandQueue : ICommandQueue
{
    public IComputeContext Context => new TestComputeContext();
    public Task FlushAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task SynchronizeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public void Dispose() { }
}

internal class TestComputeContext : IComputeContext
{
    public IComputeDevice Device => new TestComputeDevice();
    public void Dispose() { }
}