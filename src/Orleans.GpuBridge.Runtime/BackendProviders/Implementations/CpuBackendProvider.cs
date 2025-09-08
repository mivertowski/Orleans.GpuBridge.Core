using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;
using Orleans.GpuBridge.Runtime.BackendProviders.Models;

namespace Orleans.GpuBridge.Runtime.BackendProviders.Implementations;

/// <summary>
/// CPU backend provider for fallback compute operations when GPU acceleration is unavailable
/// </summary>
/// <remarks>
/// This provider serves as a fallback implementation when no GPU backend providers are available.
/// It executes compute kernels on the CPU using managed code, ensuring that applications can
/// always function even without GPU hardware or drivers. Performance will be significantly
/// lower compared to GPU implementations, but functionality is maintained.
/// </remarks>
internal class CpuBackendProvider : IBackendProvider
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger _logger;

    /// <summary>
    /// Gets the display name of this backend provider
    /// </summary>
    public string Name => "CPU";

    /// <summary>
    /// Gets the backend type identifier
    /// </summary>
    public GpuBackend Type => GpuBackend.CPU;

    /// <summary>
    /// Gets a value indicating whether this backend is currently available
    /// </summary>
    /// <remarks>
    /// Always returns true as CPU execution is always available
    /// </remarks>
    public bool IsAvailable => true;

    /// <summary>
    /// Gets the number of available CPU devices
    /// </summary>
    /// <remarks>
    /// Always returns 1 representing the single logical CPU device
    /// </remarks>
    public int DeviceCount => 1;

    /// <summary>
    /// Initializes a new instance of the <see cref="CpuBackendProvider"/> class
    /// </summary>
    /// <param name="serviceProvider">The service provider for dependency resolution</param>
    /// <param name="logger">The logger for diagnostic output</param>
    public CpuBackendProvider(IServiceProvider serviceProvider, ILogger logger)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Initializes the CPU backend provider
    /// </summary>
    /// <returns>true indicating successful initialization</returns>
    /// <remarks>
    /// Always returns true as no special initialization is required for CPU execution
    /// </remarks>
    public bool Initialize()
    {
        _logger.LogDebug("CPU backend provider initialized successfully");
        return true;
    }

    /// <summary>
    /// Shuts down the CPU backend provider and releases resources
    /// </summary>
    /// <remarks>
    /// Currently a no-op as CPU backend requires no special cleanup
    /// </remarks>
    public void Shutdown()
    {
        _logger.LogDebug("CPU backend provider shutdown");
    }

    /// <summary>
    /// Creates a compute context for CPU execution
    /// </summary>
    /// <param name="deviceIndex">The device index (ignored for CPU backend)</param>
    /// <returns>A compute context for CPU execution</returns>
    /// <exception cref="NotImplementedException">Thrown as CPU compute context is not yet implemented</exception>
    /// <remarks>
    /// This method will be implemented to create and return a CPU-specific compute context
    /// that executes kernels using managed code and threading primitives.
    /// </remarks>
    public IComputeContext CreateContext(int deviceIndex = 0)
    {
        _logger.LogInformation("Creating CPU compute context");
        return new CpuComputeContext(_logger);
    }

    /// <summary>
    /// Gets information about available CPU devices
    /// </summary>
    /// <returns>A read-only list containing information about the CPU device</returns>
    /// <remarks>
    /// Currently returns an empty list as device enumeration is not yet implemented.
    /// The real implementation will return CPU information such as core count, cache sizes,
    /// and supported instruction sets.
    /// </remarks>
    public IReadOnlyList<DeviceInfo> GetDevices()
    {
        return Array.Empty<DeviceInfo>();
    }
    
    /// <summary>
    /// Disposes resources used by this backend provider
    /// </summary>
    public void Dispose()
    {
        Shutdown();
    }
}