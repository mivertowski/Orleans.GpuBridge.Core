using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.DotCompute.Devices;

namespace Orleans.GpuBridge.DotCompute.Kernels;

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
            _device.Type != DeviceType.CPU,
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
            DeviceType.CUDA => 1024,
            DeviceType.OpenCL => 512,
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