using System;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

namespace Orleans.GpuBridge.Runtime.Providers;

/// <summary>
/// CPU compute context for fallback provider
/// </summary>
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

/// <summary>
/// CPU command queue for fallback provider
/// </summary>
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
