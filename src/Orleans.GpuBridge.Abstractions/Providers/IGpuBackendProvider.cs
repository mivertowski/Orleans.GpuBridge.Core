using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Allocators;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Interfaces;

namespace Orleans.GpuBridge.Abstractions.Providers;

/// <summary>
/// Main interface for GPU backend providers (ILGPU, DotCompute, etc.)
/// </summary>
public interface IGpuBackendProvider : IDisposable
{
    /// <summary>
    /// Unique identifier for this backend provider
    /// </summary>
    string ProviderId { get; }

    /// <summary>
    /// Display name for this backend provider
    /// </summary>
    string DisplayName { get; }

    /// <summary>
    /// Version of the backend provider
    /// </summary>
    Version Version { get; }

    /// <summary>
    /// Gets the capabilities of this backend provider
    /// </summary>
    BackendCapabilities Capabilities { get; }

    /// <summary>
    /// Initializes the backend provider
    /// </summary>
    Task InitializeAsync(BackendConfiguration configuration, CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if the backend is available on this system
    /// </summary>
    Task<bool> IsAvailableAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if the backend is available on this system (synchronous version)
    /// </summary>
    bool IsAvailable();

    /// <summary>
    /// Gets the device manager for this backend
    /// </summary>
    IDeviceManager GetDeviceManager();

    /// <summary>
    /// Gets the kernel compiler for this backend
    /// </summary>
    IKernelCompiler GetKernelCompiler();

    /// <summary>
    /// Gets the memory allocator for this backend
    /// </summary>
    IMemoryAllocator GetMemoryAllocator();

    /// <summary>
    /// Gets the kernel executor for this backend
    /// </summary>
    IKernelExecutor GetKernelExecutor();

    /// <summary>
    /// Gets backend-specific metrics
    /// </summary>
    Task<IReadOnlyDictionary<string, object>> GetMetricsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Performs a health check on the backend
    /// </summary>
    Task<HealthCheckResult> CheckHealthAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates a compute context for the specified device
    /// </summary>
    Task<object> CreateContext(int deviceIndex = 0);
}

/// <summary>
/// Configuration for a backend provider
/// </summary>
public sealed record BackendConfiguration(
    bool EnableProfiling = false,
    bool EnableDebugMode = false,
    int MaxMemoryPoolSizeMB = 2048,
    int MaxConcurrentKernels = 50,
    IReadOnlyDictionary<string, object>? CustomSettings = null);

/// <summary>
/// Health check result for a backend provider
/// </summary>
public sealed record HealthCheckResult(
    bool IsHealthy,
    string? Message = null,
    IReadOnlyDictionary<string, object>? Diagnostics = null);