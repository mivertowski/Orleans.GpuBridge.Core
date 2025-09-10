using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Enums.Compilation;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Options;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Enums;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results.Statistics;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Enums;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU fallback backend provider
/// </summary>
internal sealed class CpuFallbackProvider : IGpuBackendProvider
{
    private readonly ILogger<CpuFallbackProvider> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private CpuDeviceManager? _deviceManager;
    private CpuKernelCompiler? _kernelCompiler;
    private CpuMemoryAllocator? _memoryAllocator;
    private CpuKernelExecutor? _kernelExecutor;
    private bool _initialized;

    public string ProviderId => "CPU";
    public string DisplayName => "CPU Fallback Provider";
    public Version Version => new(1, 0, 0);
    public BackendCapabilities Capabilities => BackendCapabilities.CreateCpuFallback();

    public CpuFallbackProvider(ILogger<CpuFallbackProvider> logger, ILoggerFactory loggerFactory)
    {
        _logger = logger;
        _loggerFactory = loggerFactory;
    }

    public Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return Task.CompletedTask;

        _logger.LogInformation("Initializing CPU fallback backend provider");

        _deviceManager = new CpuDeviceManager(_loggerFactory.CreateLogger<CpuDeviceManager>());
        _kernelCompiler = new CpuKernelCompiler(_loggerFactory.CreateLogger<CpuKernelCompiler>());
        _memoryAllocator = new CpuMemoryAllocator(_loggerFactory.CreateLogger<CpuMemoryAllocator>());
        _kernelExecutor = new CpuKernelExecutor(_loggerFactory.CreateLogger<CpuKernelExecutor>());

        _initialized = true;
        return Task.CompletedTask;
    }

    public Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true); // CPU is always available
    }
    
    public bool IsAvailable()
    {
        return true; // CPU is always available
    }

    public IDeviceManager GetDeviceManager()
    {
        EnsureInitialized();
        return _deviceManager!;
    }

    public IKernelCompiler GetKernelCompiler()
    {
        EnsureInitialized();
        return _kernelCompiler!;
    }

    public IMemoryAllocator GetMemoryAllocator()
    {
        EnsureInitialized();
        return _memoryAllocator!;
    }

    public IKernelExecutor GetKernelExecutor()
    {
        EnsureInitialized();
        return _kernelExecutor!;
    }

    public Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default)
    {
        var metrics = new Dictionary<string, object>
        {
            ["provider"] = ProviderId,
            ["cpu_cores"] = Environment.ProcessorCount,
            ["memory_mb"] = GC.GetTotalMemory(false) / (1024 * 1024)
        };

        return Task.FromResult<IReadOnlyDictionary<string, object>>(metrics);
    }

    public Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new HealthCheckResult(
            IsHealthy: true,
            Message: "CPU fallback provider is healthy"));
    }

    public async Task<object> CreateContext(int deviceIndex = 0)
    {
        EnsureInitialized();
        
        if (_deviceManager == null)
            throw new InvalidOperationException("Device manager not initialized");
            
        var device = _deviceManager.GetDevice(deviceIndex);
        if (device == null)
            throw new ArgumentException($"Device at index {deviceIndex} not found", nameof(deviceIndex));
            
        var context = await _deviceManager.CreateContextAsync(device, new ContextOptions(), CancellationToken.None);
        return context;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Provider not initialized");
        }
    }

    public void Dispose()
    {
        _deviceManager?.Dispose();
        _kernelCompiler?.Dispose();
        _memoryAllocator?.Dispose();
        _kernelExecutor?.Dispose();
    }
}

// Basic CPU implementations (placeholder)
internal sealed class CpuDeviceManager : IDeviceManager
{
    private readonly ILogger<CpuDeviceManager> _logger;
    private readonly List<IComputeDevice> _devices;

    public CpuDeviceManager(ILogger<CpuDeviceManager> logger)
    {
        _logger = logger;
        _devices = new List<IComputeDevice>
        {
            new CpuDevice()
        };
    }

    public Task InitializeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public IReadOnlyList<IComputeDevice> GetDevices() => _devices;

    public IComputeDevice? GetDevice(int deviceIndex) => 
        deviceIndex == 0 ? _devices[0] : null;

    public IComputeDevice GetDefaultDevice() => _devices[0];

    public IComputeDevice SelectDevice(DeviceSelectionCriteria criteria) => _devices[0];

    public Task<IComputeContext> CreateContextAsync(
        IComputeDevice device, 
        ContextOptions options, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IComputeContext>(new CpuContext(device));
    }

    public Task<Orleans.GpuBridge.Abstractions.Models.DeviceMetrics> GetDeviceMetricsAsync(
        IComputeDevice device, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new Orleans.GpuBridge.Abstractions.Models.DeviceMetrics
        {
            GpuUtilizationPercent = 0,
            MemoryUtilizationPercent = 50,
            UsedMemoryBytes = GC.GetTotalMemory(false),
            TemperatureCelsius = 45,
            PowerWatts = 0,
            FanSpeedPercent = 0,
            KernelsExecuted = 0,
            BytesTransferred = 0,
            Uptime = TimeSpan.FromMilliseconds(Environment.TickCount64)
        });
    }

    public Task ResetDeviceAsync(
        IComputeDevice device, 
        CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose() { }
}

internal sealed class CpuDevice : IComputeDevice
{
    public string DeviceId => "cpu-0";
    public int Index => 0;
    public string Name => "CPU";
    public DeviceType Type => DeviceType.CPU;
    public string Vendor => "Generic";
    public string Architecture => "x86-64";
    public Version ComputeCapability => new(1, 0);
    public long TotalMemoryBytes => 8L * 1024 * 1024 * 1024; // 8GB
    public long AvailableMemoryBytes => 4L * 1024 * 1024 * 1024; // 4GB
    public int ComputeUnits => Environment.ProcessorCount;
    public int MaxClockFrequencyMHz => 3000;
    public int MaxThreadsPerBlock => 1024;
    public int[] MaxWorkGroupDimensions => new[] { 1024, 1024, 1024 };
    public int WarpSize => 1;
    public IReadOnlyDictionary<string, object> Properties => new Dictionary<string, object>();

    public bool SupportsFeature(string feature) => false;
    public DeviceStatus GetStatus() => DeviceStatus.Available;
}

internal sealed class CpuContext : IComputeContext
{
    public IComputeDevice Device { get; }
    public string ContextId => "cpu-context-0";

    public CpuContext(IComputeDevice device)
    {
        Device = device;
    }

    public void MakeCurrent() { }
    public Task SynchronizeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public ICommandQueue CreateCommandQueue(CommandQueueOptions options)
    {
        return new CpuCommandQueue(this);
    }

    public void Dispose() { }
}

internal sealed class CpuCommandQueue : ICommandQueue
{
    public string QueueId => "cpu-queue-0";
    public IComputeContext Context { get; }

    public CpuCommandQueue(IComputeContext context)
    {
        Context = context;
    }

    public Task EnqueueKernelAsync(
        CompiledKernel kernel, 
        KernelLaunchParameters parameters, 
        CancellationToken cancellationToken = default) => Task.CompletedTask;

    public Task EnqueueCopyAsync(
        IntPtr source, 
        IntPtr destination, 
        long sizeBytes, 
        CancellationToken cancellationToken = default) => Task.CompletedTask;

    public Task FlushAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void EnqueueBarrier() { }

    public void Dispose() { }
}

internal sealed class CpuKernelCompiler : IKernelCompiler
{
    private readonly ILogger<CpuKernelCompiler> _logger;

    public CpuKernelCompiler(ILogger<CpuKernelCompiler> logger)
    {
        _logger = logger;
    }

    [RequiresUnreferencedCode("Uses method validation which may not work with trimming.")]
    public Task<CompiledKernel> CompileFromMethodAsync(
        [NotNull] System.Reflection.MethodInfo method, 
        [NotNull] KernelCompilationOptions options, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompiledKernel
        {
            KernelId = method.Name,
            Name = method.Name,
            CompiledCode = Array.Empty<byte>(),
            Metadata = new KernelMetadata()
        });
    }

    public Task<CompiledKernel> CompileFromSourceAsync(
        [NotNull] string sourceCode, 
        [NotNull] string entryPoint, 
        KernelLanguage language, 
        [NotNull] KernelCompilationOptions options, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompiledKernel
        {
            KernelId = entryPoint,
            Name = entryPoint,
            CompiledCode = System.Text.Encoding.UTF8.GetBytes(sourceCode),
            Metadata = new KernelMetadata()
        });
    }

    [RequiresUnreferencedCode("Uses reflection to find types and methods which may be trimmed.")]
    public Task<CompiledKernel> CompileFromAssemblyAsync(
        [NotNull] System.Reflection.Assembly assembly, 
        [NotNull] string typeName, 
        [NotNull] string methodName, 
        [NotNull] KernelCompilationOptions options, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompiledKernel
        {
            KernelId = $"{typeName}.{methodName}",
            Name = methodName,
            CompiledCode = Array.Empty<byte>(),
            Metadata = new KernelMetadata()
        });
    }

    [RequiresUnreferencedCode("Uses method body analysis which may not work with trimming.")]
    public Task<KernelValidationResult> ValidateMethodAsync(
        [NotNull] System.Reflection.MethodInfo method, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new KernelValidationResult(IsValid: true));
    }

    public Task<CompilationDiagnostics> GetDiagnosticsAsync(
        [NotNull] CompiledKernel kernel, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new CompilationDiagnostics());
    }

    public void ClearCache() { }

    public void Dispose() { }
}

internal sealed class CpuMemoryAllocator : IMemoryAllocator
{
    private readonly ILogger<CpuMemoryAllocator> _logger;

    public CpuMemoryAllocator(ILogger<CpuMemoryAllocator> logger)
    {
        _logger = logger;
    }

    public Task<IDeviceMemory> AllocateAsync(
        long sizeBytes, 
        MemoryAllocationOptions options, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IDeviceMemory>(new CpuMemory(sizeBytes));
    }

    public Task<IDeviceMemory<T>> AllocateAsync<T>(
        int elementCount, 
        MemoryAllocationOptions options, 
        CancellationToken cancellationToken = default) where T : unmanaged
    {
        return Task.FromResult<IDeviceMemory<T>>(new CpuMemory<T>(elementCount));
    }

    public Task<IPinnedMemory> AllocatePinnedAsync(
        long sizeBytes, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IPinnedMemory>(new CpuPinnedMemory(sizeBytes));
    }

    public Task<IUnifiedMemory> AllocateUnifiedAsync(
        long sizeBytes, 
        UnifiedMemoryOptions options, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IUnifiedMemory>(new CpuUnifiedMemory(sizeBytes));
    }

    public MemoryPoolStatistics GetPoolStatistics()
    {
        return new MemoryPoolStatistics(
            TotalBytesAllocated: 0,
            TotalBytesInUse: 0,
            TotalBytesFree: long.MaxValue,
            AllocationCount: 0,
            FreeBlockCount: 1,
            LargestFreeBlock: long.MaxValue,
            FragmentationPercent: 0,
            PeakUsageBytes: 0);
    }

    public Task CompactAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task ResetAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose() { }
}

// Placeholder CPU memory classes
internal sealed class CpuMemory : IDeviceMemory
{
    public long SizeBytes { get; }
    public IntPtr DevicePointer { get; }
    public IComputeDevice Device => new CpuDevice();

    public CpuMemory(long sizeBytes)
    {
        SizeBytes = sizeBytes;
        DevicePointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)sizeBytes);
    }

    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) => new CpuMemory(sizeBytes);

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(DevicePointer);
    }
}

internal sealed class CpuMemory<T> : IDeviceMemory<T> where T : unmanaged
{
    public int Length { get; }
    public long SizeBytes { get; }
    public IntPtr DevicePointer { get; }
    public IComputeDevice Device { get; }

    public CpuMemory(int elementCount)
    {
        Length = elementCount;
        unsafe { SizeBytes = elementCount * sizeof(T); }
        DevicePointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)SizeBytes);
        Device = new CpuDevice();
    }

    public unsafe Span<T> AsSpan()
    {
        return new Span<T>((void*)DevicePointer, Length);
    }

    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) 
    { 
        unsafe { return new CpuMemory<T>((int)(sizeBytes / sizeof(T))); }
    }

    public Task CopyFromHostAsync(T[] source, int sourceOffset, int destinationOffset, int count, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(T[] destination, int sourceOffset, int destinationOffset, int count, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(T value, int offset, int count, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(DevicePointer);
    }
}

internal sealed class CpuPinnedMemory : IPinnedMemory
{
    public long SizeBytes { get; }
    public IntPtr HostPointer { get; }

    public CpuPinnedMemory(long sizeBytes)
    {
        SizeBytes = sizeBytes;
        HostPointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)sizeBytes);
    }

    public unsafe Span<byte> AsSpan()
    {
        return new Span<byte>((void*)HostPointer, (int)SizeBytes);
    }

    public Task RegisterWithDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task UnregisterFromDeviceAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(HostPointer);
    }
}

internal sealed class CpuUnifiedMemory : IUnifiedMemory
{
    public long SizeBytes { get; }
    public IntPtr DevicePointer { get; }
    public IComputeDevice Device { get; }
    public IntPtr HostPointer => DevicePointer;

    public CpuUnifiedMemory(long sizeBytes)
    {
        SizeBytes = sizeBytes;
        DevicePointer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)sizeBytes);
        Device = new CpuDevice();
    }

    public Task CopyFromHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyToHostAsync(IntPtr hostPointer, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task CopyFromAsync(IDeviceMemory source, long sourceOffset, long destinationOffset, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task FillAsync(byte value, long offsetBytes, long sizeBytes, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public IDeviceMemory CreateView(long offsetBytes, long sizeBytes) => new CpuUnifiedMemory(sizeBytes);

    public Task PrefetchAsync(IComputeDevice device, CancellationToken cancellationToken = default) => Task.CompletedTask;
    public Task AdviseAsync(MemoryAdvice advice, IComputeDevice? device = null, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public unsafe Span<byte> AsHostSpan()
    {
        return new Span<byte>((void*)HostPointer, (int)SizeBytes);
    }

    public void Dispose()
    {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(DevicePointer);
    }
}

internal sealed class CpuKernelExecutor : IKernelExecutor
{
    private readonly ILogger<CpuKernelExecutor> _logger;

    public CpuKernelExecutor(ILogger<CpuKernelExecutor> logger)
    {
        _logger = logger;
    }

    public Task<KernelExecutionResult> ExecuteAsync(
        CompiledKernel kernel, 
        KernelExecutionParameters parameters, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new KernelExecutionResult(Success: true));
    }

    public Task<IKernelExecution> ExecuteAsyncNonBlocking(
        CompiledKernel kernel, 
        KernelExecutionParameters parameters, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IKernelExecution>(new CpuKernelExecution(kernel));
    }

    public Task<BatchExecutionResult> ExecuteBatchAsync(
        IReadOnlyList<KernelBatchItem> batch, 
        BatchExecutionOptions options, 
        CancellationToken cancellationToken = default)
    {
        var results = batch.Select(_ => new KernelExecutionResult(Success: true)).ToList();
        return Task.FromResult(new BatchExecutionResult(
            SuccessCount: batch.Count,
            FailureCount: 0,
            Results: results,
            TotalExecutionTime: TimeSpan.Zero));
    }

    public IKernelGraph CreateGraph(string graphName)
    {
        return new CpuKernelGraph(graphName);
    }

    public Task<KernelProfile> ProfileAsync(
        CompiledKernel kernel, 
        KernelExecutionParameters parameters, 
        int iterations = 100, 
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new KernelProfile(
            AverageExecutionTime: TimeSpan.FromMilliseconds(1),
            MinExecutionTime: TimeSpan.FromMilliseconds(0.5),
            MaxExecutionTime: TimeSpan.FromMilliseconds(2),
            StandardDeviation: 0.1,
            MemoryBandwidthBytesPerSecond: 0,
            ComputeThroughputGFlops: 0,
            OptimalBlockSize: 256));
    }

    public ExecutionStatistics GetStatistics()
    {
        return new ExecutionStatistics(
            TotalKernelsExecuted: 0,
            TotalBatchesExecuted: 0,
            TotalGraphsExecuted: 0,
            TotalExecutionTime: TimeSpan.Zero,
            AverageKernelTime: TimeSpan.Zero,
            TotalBytesTransferred: 0,
            TotalErrors: 0,
            KernelExecutionCounts: new Dictionary<string, long>());
    }

    public void ResetStatistics() { }

    public void Dispose() { }
}

internal sealed class CpuKernelExecution : IKernelExecution
{
    public string ExecutionId => Guid.NewGuid().ToString();
    public CompiledKernel Kernel { get; }
    public KernelExecutionStatus Status => KernelExecutionStatus.Completed;
    public bool IsComplete => true;
    public double Progress => 1.0;

    public CpuKernelExecution(CompiledKernel kernel)
    {
        Kernel = kernel;
    }

    public Task<KernelExecutionResult> WaitForCompletionAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new KernelExecutionResult(Success: true));
    }

    public Task CancelAsync() => Task.CompletedTask;

    public KernelTiming? GetTiming()
    {
        return new KernelTiming(TimeSpan.Zero, TimeSpan.FromMilliseconds(1), TimeSpan.FromMilliseconds(1));
    }
}

internal sealed class CpuKernelGraph : IKernelGraph
{
    public string Name { get; }

    public CpuKernelGraph(string name)
    {
        Name = name;
    }

    public IGraphNode AddKernel(CompiledKernel kernel, KernelExecutionParameters parameters, IReadOnlyList<IGraphNode>? dependencies = null)
    {
        return new CpuGraphNode("kernel-" + Guid.NewGuid().ToString(), GraphNodeType.Kernel, dependencies ?? Array.Empty<IGraphNode>());
    }

    public IGraphNode AddMemCopy(IDeviceMemory source, IDeviceMemory destination, long sizeBytes, IReadOnlyList<IGraphNode>? dependencies = null)
    {
        return new CpuGraphNode("memcpy-" + Guid.NewGuid().ToString(), GraphNodeType.MemCopy, dependencies ?? Array.Empty<IGraphNode>());
    }

    public IGraphNode AddBarrier(IReadOnlyList<IGraphNode> dependencies)
    {
        return new CpuGraphNode("barrier-" + Guid.NewGuid().ToString(), GraphNodeType.Barrier, dependencies);
    }

    public Task<ICompiledGraph> CompileAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult<ICompiledGraph>(new CpuCompiledGraph());
    }

    public GraphValidationResult Validate()
    {
        return new GraphValidationResult(IsValid: true);
    }

    public void Dispose() { }
}

internal sealed class CpuGraphNode : IGraphNode
{
    public string NodeId { get; }
    public GraphNodeType Type { get; }
    public IReadOnlyList<IGraphNode> Dependencies { get; }

    public CpuGraphNode(string nodeId, GraphNodeType type, IReadOnlyList<IGraphNode> dependencies)
    {
        NodeId = nodeId;
        Type = type;
        Dependencies = dependencies;
    }
}

internal sealed class CpuCompiledGraph : ICompiledGraph
{
    public Task<GraphExecutionResult> ExecuteAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new GraphExecutionResult(
            Success: true,
            NodesExecuted: 0,
            ExecutionTime: TimeSpan.Zero));
    }

    public void UpdateParameters(string nodeId, KernelExecutionParameters parameters) { }

    public void Dispose() { }
}