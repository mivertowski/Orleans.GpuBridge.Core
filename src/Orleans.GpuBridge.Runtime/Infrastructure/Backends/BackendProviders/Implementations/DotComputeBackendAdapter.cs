using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;
using Orleans.GpuBridge.Runtime.BackendProviders.Models;
using RuntimeIComputeContext = Orleans.GpuBridge.Runtime.BackendProviders.Interfaces.IComputeContext;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Implementations;

/// <summary>
/// Adapter that wraps DotComputeBackendProvider to implement the IBackendProvider interface.
/// Routes all GPU operations (CUDA, OpenCL, Vulkan, Metal, DirectCompute) through DotCompute.
/// </summary>
/// <remarks>
/// DotCompute provides a unified abstraction over multiple GPU backends:
/// - CUDA (NVIDIA GPUs)
/// - OpenCL (Cross-platform)
/// - Vulkan Compute (Cross-platform)
/// - Metal (macOS/iOS)
/// - CPU fallback (always available)
///
/// This adapter bridges the Runtime's IBackendProvider interface with the
/// Abstractions' IGpuBackendProvider interface used by DotCompute.
/// </remarks>
internal sealed class DotComputeBackendAdapter : IBackendProvider
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger _logger;
    private IGpuBackendProvider? _dotComputeProvider;
    private bool _isInitialized;

    /// <summary>
    /// Gets the display name of this backend provider.
    /// </summary>
    public string Name => "DotCompute";

    /// <summary>
    /// Gets the backend type identifier.
    /// Reports as CUDA since DotCompute primarily targets NVIDIA GPUs via CUDA,
    /// but transparently supports other backends.
    /// </summary>
    public GpuBackend Type => GpuBackend.CUDA;

    /// <summary>
    /// Gets a value indicating whether this backend is currently available.
    /// </summary>
    public bool IsAvailable
    {
        get
        {
            if (!_isInitialized || _dotComputeProvider == null)
            {
                return false;
            }

            return _dotComputeProvider.IsAvailable();
        }
    }

    /// <summary>
    /// Gets the number of available compute devices.
    /// </summary>
    public int DeviceCount
    {
        get
        {
            if (!_isInitialized || _dotComputeProvider == null)
            {
                return 0;
            }

            try
            {
                var deviceManager = _dotComputeProvider.GetDeviceManager();
                return deviceManager.GetDevices().Count;
            }
            catch
            {
                return 0;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DotComputeBackendAdapter"/> class.
    /// </summary>
    /// <param name="serviceProvider">The service provider for dependency resolution.</param>
    /// <param name="logger">The logger for diagnostic output.</param>
    public DotComputeBackendAdapter(IServiceProvider serviceProvider, ILogger logger)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Initializes the DotCompute backend provider.
    /// </summary>
    /// <returns>true if initialization was successful; otherwise, false.</returns>
    public bool Initialize()
    {
        if (_isInitialized)
        {
            _logger.LogDebug("DotCompute backend adapter already initialized");
            return IsAvailable;
        }

        try
        {
            // Try to resolve DotComputeBackendProvider from DI
            _dotComputeProvider = _serviceProvider.GetService<IGpuBackendProvider>();

            if (_dotComputeProvider == null)
            {
                _logger.LogDebug(
                    "DotCompute backend provider not registered in DI. " +
                    "Register with services.AddDotComputeBackend() to enable GPU acceleration.");
                return false;
            }

            // DotComputeBackendProvider requires async initialization
            // For IBackendProvider sync interface, we check if it's already initialized
            if (_dotComputeProvider.IsAvailable())
            {
                _isInitialized = true;
                var deviceCount = DeviceCount;

                _logger.LogInformation(
                    "DotCompute backend adapter initialized successfully ({DeviceCount} device(s) available)",
                    deviceCount);

                return true;
            }

            // Provider exists but not yet initialized - initialization happens lazily
            // when the provider is first used through GpuBridge services
            _isInitialized = true;
            _logger.LogDebug(
                "DotCompute backend adapter registered. Provider will initialize on first use.");

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to initialize DotCompute backend adapter");
            return false;
        }
    }

    /// <summary>
    /// Shuts down the backend provider and releases resources.
    /// </summary>
    public void Shutdown()
    {
        _logger.LogDebug("DotCompute backend adapter shutdown");
        _isInitialized = false;
        // DotComputeBackendProvider disposal is handled by DI container
    }

    /// <summary>
    /// Creates a compute context for the specified device.
    /// </summary>
    /// <param name="deviceIndex">The zero-based index of the device.</param>
    /// <returns>A compute context for the specified device.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the provider is not initialized.</exception>
    public RuntimeIComputeContext CreateContext(int deviceIndex = 0)
    {
        if (!_isInitialized || _dotComputeProvider == null)
        {
            throw new InvalidOperationException(
                "DotCompute backend adapter is not initialized. Call Initialize() first.");
        }

        if (!IsAvailable)
        {
            throw new InvalidOperationException(
                "DotCompute backend is not available. No compute devices found.");
        }

        try
        {
            _logger.LogDebug("Creating DotCompute compute context for device index {DeviceIndex}", deviceIndex);

            // Create context asynchronously and wrap in sync call
            // This is a bridge pattern - in production, prefer async APIs
            var contextTask = _dotComputeProvider.CreateContext(deviceIndex);
            var context = contextTask.GetAwaiter().GetResult();

            return new DotComputeContextWrapper(context, _logger, deviceIndex);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to create DotCompute compute context for device {DeviceIndex}", deviceIndex);
            throw;
        }
    }

    /// <summary>
    /// Gets information about all available compute devices.
    /// </summary>
    /// <returns>A read-only list of device information.</returns>
    public IReadOnlyList<DeviceInfo> GetDevices()
    {
        if (!_isInitialized || _dotComputeProvider == null)
        {
            return Array.Empty<DeviceInfo>();
        }

        try
        {
            var deviceManager = _dotComputeProvider.GetDeviceManager();
            var devices = deviceManager.GetDevices();
            var result = new List<DeviceInfo>();

            int index = 0;
            foreach (var device in devices)
            {
                result.Add(new DeviceInfo(
                    Index: index++,
                    Name: device.Name,
                    Backend: GpuBackend.CUDA, // DotCompute primarily uses CUDA
                    TotalMemory: device.TotalMemoryBytes,
                    ComputeUnits: device.MaxThreadsPerBlock,
                    Extensions: Array.Empty<string>()
                ));
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to enumerate DotCompute devices");
            return Array.Empty<DeviceInfo>();
        }
    }

    /// <summary>
    /// Disposes resources used by this backend adapter.
    /// </summary>
    public void Dispose()
    {
        Shutdown();
    }
}

/// <summary>
/// Wrapper that adapts DotCompute context objects to the IComputeContext interface.
/// </summary>
internal sealed class DotComputeContextWrapper : RuntimeIComputeContext
{
    private readonly object _dotComputeContext;
    private readonly ILogger _logger;
    private readonly int _deviceIndex;
    private bool _disposed;

    /// <summary>
    /// Gets the GPU backend type.
    /// </summary>
    public GpuBackend Backend => GpuBackend.CUDA;

    /// <summary>
    /// Gets the device index this context is associated with.
    /// </summary>
    public int DeviceIndex => _deviceIndex;

    public DotComputeContextWrapper(object dotComputeContext, ILogger logger, int deviceIndex = 0)
    {
        _dotComputeContext = dotComputeContext ?? throw new ArgumentNullException(nameof(dotComputeContext));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _deviceIndex = deviceIndex;
    }

    /// <summary>
    /// Creates a compute buffer with the specified size and usage.
    /// </summary>
    public IComputeBuffer<T> CreateBuffer<T>(int size, BufferUsage usage) where T : unmanaged
    {
        ThrowIfDisposed();

        _logger.LogDebug(
            "Creating compute buffer: Type={Type}, Size={Size}, Usage={Usage}",
            typeof(T).Name, size, usage);

        // DotCompute context operations would be delegated here
        // For now, throw not supported as actual DotCompute integration happens at higher level
        throw new NotSupportedException(
            "Direct buffer creation through DotCompute context wrapper is not supported. " +
            "Use DotComputeDeviceMemory for GPU memory operations.");
    }

    /// <summary>
    /// Compiles a kernel from source code.
    /// </summary>
    public IComputeKernel CompileKernel(string source, string entryPoint)
    {
        ThrowIfDisposed();

        _logger.LogDebug("Compiling kernel: EntryPoint={EntryPoint}", entryPoint);

        // DotCompute context operations would be delegated here
        throw new NotSupportedException(
            "Direct kernel compilation through DotCompute context wrapper is not supported. " +
            "Use RingKernel infrastructure for GPU kernel execution.");
    }

    /// <summary>
    /// Executes a compiled kernel.
    /// </summary>
    public void Execute(IComputeKernel kernel, int workSize)
    {
        ThrowIfDisposed();

        _logger.LogDebug("Executing kernel: WorkSize={WorkSize}", workSize);

        // DotCompute context operations would be delegated here
        throw new NotSupportedException(
            "Direct kernel execution through DotCompute context wrapper is not supported. " +
            "Use RingKernel infrastructure for GPU kernel execution.");
    }

    /// <summary>
    /// Synchronizes the compute context (waits for all pending operations).
    /// </summary>
    public void Synchronize()
    {
        ThrowIfDisposed();

        _logger.LogTrace("Synchronizing DotCompute context");

        // DotCompute synchronization would be called here
        // For managed contexts, this is typically a no-op
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            if (_dotComputeContext is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error disposing DotCompute context");
        }
        finally
        {
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
