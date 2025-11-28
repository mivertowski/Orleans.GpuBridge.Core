// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// Wraps DotCompute's implicit context management for Orleans.GpuBridge compatibility.
/// </summary>
/// <remarks>
/// <para>
/// DotCompute v0.5.1 uses implicit context management via <see cref="IAccelerator.Context"/>.
/// This wrapper provides the <see cref="IComputeContext"/> interface expected by Orleans.GpuBridge
/// while delegating to DotCompute's internal context handling.
/// </para>
/// <para>
/// Context lifecycle is managed automatically by DotCompute - this wrapper primarily
/// tracks Orleans-side state and provides synchronization points.
/// </para>
/// </remarks>
internal sealed class DotComputeImplicitContext : IComputeContext
{
    private readonly IAccelerator _accelerator;
    private readonly IComputeDevice _device;
    private readonly string _contextId;
    private bool _disposed;

    /// <summary>
    /// Creates a new implicit context wrapper.
    /// </summary>
    /// <param name="device">The Orleans.GpuBridge device wrapper.</param>
    /// <param name="accelerator">The DotCompute accelerator with implicit context.</param>
    public DotComputeImplicitContext(
        IComputeDevice device,
        IAccelerator accelerator)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _contextId = $"dotcompute-ctx-{device.DeviceId}-{Guid.NewGuid():N}";
    }

    /// <inheritdoc />
    public IComputeDevice Device => _device;

    /// <inheritdoc />
    public string ContextId => _contextId;

    /// <summary>
    /// Gets the device name.
    /// </summary>
    public string DeviceName => _device.Name;

    /// <summary>
    /// Gets whether this context is active and not disposed.
    /// </summary>
    public bool IsActive => !_disposed && _accelerator.IsAvailable;

    /// <inheritdoc />
    public void MakeCurrent()
    {
        ThrowIfDisposed();
        // DotCompute uses implicit context management per accelerator.
        // Making context "current" is effectively selecting which accelerator to use.
        // Since each context is bound to a specific accelerator, this is a no-op
        // as DotCompute handles context internally.
    }

    /// <inheritdoc />
    public async Task SynchronizeAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        await _accelerator.SynchronizeAsync(cancellationToken);
    }

    /// <inheritdoc />
    public ICommandQueue CreateCommandQueue(CommandQueueOptions options)
    {
        ThrowIfDisposed();
        return new DotComputeCommandQueue(this, _accelerator, options);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;

        // Synchronize to ensure all operations complete before disposing
        // Use a short timeout to avoid blocking indefinitely
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            _accelerator.SynchronizeAsync(cts.Token).GetAwaiter().GetResult();
        }
        catch
        {
            // Ignore sync errors during disposal
        }

        _disposed = true;
    }

    /// <summary>
    /// Gets the underlying DotCompute accelerator for advanced operations.
    /// </summary>
    internal IAccelerator Accelerator => _accelerator;

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
