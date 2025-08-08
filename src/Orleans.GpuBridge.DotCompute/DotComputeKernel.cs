using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.DotCompute;

/// <summary>
/// DotCompute kernel implementation that provides GPU acceleration
/// </summary>
public abstract class DotComputeKernel<TIn, TOut> : IGpuKernel<TIn, TOut>
    where TIn : notnull
    where TOut : notnull
{
    protected readonly ILogger<DotComputeKernel<TIn, TOut>> _logger;
    protected readonly IComputeDevice _device;
    protected readonly KernelId _kernelId;
    protected readonly string _kernelCode;
    private readonly SemaphoreSlim _executionLock;
    private readonly Dictionary<string, KernelExecutionContext> _activeHandles;
    
    protected DotComputeKernel(
        KernelId kernelId,
        IComputeDevice device,
        string kernelCode,
        ILogger<DotComputeKernel<TIn, TOut>> logger)
    {
        _kernelId = kernelId;
        _device = device;
        _kernelCode = kernelCode;
        _logger = logger;
        _executionLock = new SemaphoreSlim(1, 1);
        _activeHandles = new Dictionary<string, KernelExecutionContext>();
    }
    
    public virtual async ValueTask<KernelHandle> SubmitBatchAsync(
        IReadOnlyList<TIn> items,
        GpuExecutionHints? hints = null,
        CancellationToken ct = default)
    {
        var handle = KernelHandle.Create();
        
        await _executionLock.WaitAsync(ct);
        try
        {
            var context = new KernelExecutionContext
            {
                Handle = handle,
                InputData = items,
                Results = new List<TOut>(),
                StartTime = DateTime.UtcNow,
                Hints = hints
            };
            
            _activeHandles[handle.Id] = context;
            
            // Start async execution
            _ = ExecuteKernelAsync(context, ct);
            
            _logger.LogDebug(
                "Submitted batch of {Count} items to kernel {KernelId}, handle {Handle}",
                items.Count, _kernelId, handle.Id);
            
            return handle;
        }
        finally
        {
            _executionLock.Release();
        }
    }
    
    public virtual async IAsyncEnumerable<TOut> ReadResultsAsync(
        KernelHandle handle,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!_activeHandles.TryGetValue(handle.Id, out var context))
        {
            throw new ArgumentException($"Handle {handle.Id} not found");
        }
        
        // Wait for execution to complete
        while (!context.IsComplete && !ct.IsCancellationRequested)
        {
            await Task.Delay(10, ct);
        }
        
        // Stream results
        foreach (var result in context.Results)
        {
            yield return result;
        }
        
        // Clean up
        _activeHandles.Remove(handle.Id);
    }
    
    public virtual ValueTask<KernelInfo> GetInfoAsync(CancellationToken ct = default)
    {
        var info = new KernelInfo(
            _kernelId,
            $"DotCompute kernel: {_kernelId}",
            typeof(TIn),
            typeof(TOut),
            _device.Type != DeviceType.Cpu,
            GetPreferredBatchSize());
        
        return new ValueTask<KernelInfo>(info);
    }
    
    protected abstract Task ExecuteKernelAsync(
        KernelExecutionContext context,
        CancellationToken ct);
    
    protected virtual int GetPreferredBatchSize()
    {
        return _device.Type switch
        {
            DeviceType.Cuda => 1024,
            DeviceType.OpenCl => 512,
            DeviceType.DirectCompute => 256,
            DeviceType.Metal => 512,
            _ => 64
        };
    }
    
    protected class KernelExecutionContext
    {
        public KernelHandle Handle { get; init; } = default!;
        public IReadOnlyList<TIn> InputData { get; init; } = default!;
        public List<TOut> Results { get; init; } = default!;
        public DateTime StartTime { get; init; }
        public GpuExecutionHints? Hints { get; init; }
        public bool IsComplete { get; set; }
        public Exception? Error { get; set; }
    }
}

/// <summary>
/// Compute device abstraction for DotCompute
/// </summary>
public interface IComputeDevice
{
    string Name { get; }
    DeviceType Type { get; }
    int Index { get; }
    long TotalMemory { get; }
    long AvailableMemory { get; }
    int ComputeUnits { get; }
    bool IsAvailable { get; }
    
    Task<IUnifiedBuffer<T>> AllocateBufferAsync<T>(
        int size,
        BufferFlags flags = BufferFlags.ReadWrite,
        CancellationToken ct = default) where T : unmanaged;
    
    Task<ICompiledKernel> CompileKernelAsync(
        string code,
        string entryPoint,
        CompilationOptions? options = null,
        CancellationToken ct = default);
    
    Task<IKernelExecution> LaunchKernelAsync(
        ICompiledKernel kernel,
        KernelLaunchParams launchParams,
        CancellationToken ct = default);
}

/// <summary>
/// Unified buffer for efficient memory management
/// </summary>
public interface IUnifiedBuffer<T> : IDisposable where T : unmanaged
{
    int Length { get; }
    Memory<T> Memory { get; }
    bool IsResident { get; }
    
    Task CopyToDeviceAsync(CancellationToken ct = default);
    Task CopyFromDeviceAsync(CancellationToken ct = default);
    Task<IUnifiedBuffer<T>> CloneAsync(CancellationToken ct = default);
}

/// <summary>
/// Compiled kernel ready for execution
/// </summary>
public interface ICompiledKernel : IDisposable
{
    string Name { get; }
    IComputeDevice Device { get; }
    
    void SetBuffer(int index, IUnifiedBuffer<byte> buffer);
    void SetConstant<T>(string name, T value) where T : unmanaged;
}

/// <summary>
/// Kernel execution handle
/// </summary>
public interface IKernelExecution
{
    bool IsComplete { get; }
    Task WaitForCompletionAsync(CancellationToken ct = default);
    TimeSpan GetExecutionTime();
}

/// <summary>
/// Buffer allocation flags
/// </summary>
[Flags]
public enum BufferFlags
{
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = ReadOnly | WriteOnly,
    HostVisible = 4,
    DeviceLocal = 8,
    Pinned = 16
}

/// <summary>
/// Kernel compilation options
/// </summary>
public sealed class CompilationOptions
{
    public bool EnableOptimizations { get; set; } = true;
    public bool EnableDebugInfo { get; set; }
    public string? TargetArchitecture { get; set; }
    public Dictionary<string, string> Defines { get; set; } = new();
    public int MaxRegisters { get; set; }
    public bool UseFastMath { get; set; } = true;
}

/// <summary>
/// Kernel launch parameters
/// </summary>
public sealed class KernelLaunchParams
{
    public int GlobalWorkSize { get; set; }
    public int LocalWorkSize { get; set; } = 256;
    public int SharedMemoryBytes { get; set; }
    public Dictionary<int, IUnifiedBuffer<byte>> Buffers { get; set; } = new();
    public Dictionary<string, object> Constants { get; set; } = new();
}