// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Models.Compilation;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// Command queue implementation wrapping DotCompute's stream-based execution model.
/// </summary>
/// <remarks>
/// <para>
/// DotCompute v0.5.1 uses an implicit command stream model where operations are
/// submitted directly to the accelerator. This wrapper provides the explicit
/// <see cref="ICommandQueue"/> interface expected by Orleans.GpuBridge.
/// </para>
/// <para>
/// Operations are tracked and synchronized through the underlying accelerator's
/// synchronization primitives. Profiling support is enabled through DotCompute's
/// timing infrastructure when requested via <see cref="CommandQueueOptions"/>.
/// </para>
/// </remarks>
internal sealed class DotComputeCommandQueue : ICommandQueue
{
    private readonly DotComputeImplicitContext _context;
    private readonly IAccelerator _accelerator;
    private readonly CommandQueueOptions _options;
    private readonly string _queueId;
    private bool _disposed;

    /// <summary>
    /// Creates a new command queue wrapper.
    /// </summary>
    /// <param name="context">The parent compute context.</param>
    /// <param name="accelerator">The DotCompute accelerator for execution.</param>
    /// <param name="options">Queue configuration options.</param>
    public DotComputeCommandQueue(
        DotComputeImplicitContext context,
        IAccelerator accelerator,
        CommandQueueOptions options)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _options = options;
        var shortGuid = Guid.NewGuid().ToString("N")[..8];
        _queueId = $"dotcompute-queue-{shortGuid}";
    }

    /// <inheritdoc />
    public string QueueId => _queueId;

    /// <inheritdoc />
    public IComputeContext Context => _context;

    /// <inheritdoc />
    public async Task EnqueueKernelAsync(
        CompiledKernel kernel,
        KernelLaunchParameters parameters,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        // DotCompute v0.5.1 kernel execution is handled through IKernelExecutor
        // which is configured at the device manager level. The kernel adapter
        // contains the necessary DotCompute references for execution.
        //
        // For implicit command queue mode, we delegate to the accelerator's
        // async execution model. The actual kernel launch is coordinated
        // through the kernel executor service.

        // Validate kernel has DotCompute adapter
        if (kernel.BackendData is not IDotComputeKernelAdapter dcKernel)
        {
            throw new InvalidOperationException(
                $"Kernel '{kernel.KernelId}' does not have a valid DotCompute backend adapter. " +
                "Ensure the kernel was compiled using DotComputeKernelCompiler.");
        }

        // Execute through the DotCompute kernel adapter
        await dcKernel.ExecuteAsync(parameters, cancellationToken);
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// DotCompute v0.5.1 uses typed memory operations with DeviceMemory handles from
    /// DotCompute.Abstractions.Results namespace. For raw IntPtr-based copies, this
    /// implementation uses managed memory copy as a fallback.
    /// </para>
    /// <para>
    /// For optimal GPU memory transfers, use the typed memory buffer APIs directly through the
    /// device memory allocator which provides proper host-device and device-device copy support.
    /// </para>
    /// </remarks>
    public Task EnqueueCopyAsync(
        IntPtr source,
        IntPtr destination,
        long sizeBytes,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        // DotCompute v0.5.1 uses DeviceMemory handles for GPU memory operations,
        // not raw IntPtr. For raw pointer copies, we use unsafe managed copy.
        // This is a fallback - typed buffer operations are preferred for GPU memory.
        unsafe
        {
            Buffer.MemoryCopy(
                source.ToPointer(),
                destination.ToPointer(),
                sizeBytes,
                sizeBytes);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public async Task FlushAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        // Synchronize all pending operations on this accelerator
        await _accelerator.SynchronizeAsync(cancellationToken);
    }

    /// <inheritdoc />
    public void EnqueueBarrier()
    {
        ThrowIfDisposed();

        // DotCompute implicit model: barriers are synchronization points
        // Force synchronization to ensure all prior operations complete
        // Note: This is a blocking operation in the implicit model
        _accelerator.SynchronizeAsync().GetAwaiter().GetResult();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;

        // Flush any pending operations before disposal
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            FlushAsync(cts.Token).GetAwaiter().GetResult();
        }
        catch
        {
            // Ignore flush errors during disposal
        }

        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}

/// <summary>
/// Marker interface for DotCompute kernel adapters.
/// </summary>
/// <remarks>
/// This interface is implemented by compiled kernel wrappers that contain
/// the necessary DotCompute references for GPU execution. It provides
/// the execution bridge between Orleans.GpuBridge's kernel abstraction
/// and DotCompute's runtime execution model.
/// </remarks>
internal interface IDotComputeKernelAdapter
{
    /// <summary>
    /// Executes the kernel with the specified launch parameters.
    /// </summary>
    /// <param name="parameters">Launch configuration including grid/block dimensions.</param>
    /// <param name="cancellationToken">Cancellation token for the operation.</param>
    /// <returns>A task representing the asynchronous kernel execution.</returns>
    Task ExecuteAsync(KernelLaunchParameters parameters, CancellationToken cancellationToken);
}
