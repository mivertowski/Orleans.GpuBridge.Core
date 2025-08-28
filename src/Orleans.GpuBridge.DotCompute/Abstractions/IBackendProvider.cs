using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.DotCompute.Devices;

namespace Orleans.GpuBridge.DotCompute.Abstractions;

/// <summary>
/// Defines the contract for compute backend providers that manage specific GPU/CPU compute APIs.
/// </summary>
/// <remarks>
/// Backend providers are responsible for initializing and managing access to specific compute
/// platforms such as CUDA, OpenCL, DirectCompute, or Metal. Each provider encapsulates the
/// platform-specific details of device discovery, initialization, and resource management.
/// Implementations should handle platform availability checks and graceful degradation
/// when the underlying compute runtime is not available.
/// </remarks>
public interface IBackendProvider : IDisposable
{
    /// <summary>
    /// Gets the human-readable name of the compute backend.
    /// </summary>
    /// <value>
    /// A descriptive name for the backend (e.g., "CUDA", "OpenCL", "DirectCompute", "Metal").
    /// </value>
    /// <remarks>
    /// This name is used for logging, diagnostics, and user-facing displays of the
    /// currently active compute backend.
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Determines whether the compute backend is available on the current platform.
    /// </summary>
    /// <returns>
    /// <c>true</c> if the backend runtime libraries are available and the platform
    /// supports this compute backend; otherwise, <c>false</c>.
    /// </returns>
    /// <remarks>
    /// This method should check for the presence of required runtime libraries,
    /// platform compatibility, and any necessary drivers or system components.
    /// It should not perform heavy initialization work and should be safe to call
    /// multiple times.
    /// </remarks>
    bool IsAvailable();

    /// <summary>
    /// Initializes the compute backend and prepares it for device enumeration and use.
    /// </summary>
    /// <param name="ct">A cancellation token to observe while waiting for the task to complete.</param>
    /// <returns>A task that represents the asynchronous initialization operation.</returns>
    /// <remarks>
    /// This method should perform any necessary initialization of the compute backend,
    /// such as loading runtime libraries, initializing the compute context, or
    /// setting up backend-specific resources. This method will only be called if
    /// <see cref="IsAvailable"/> returns <c>true</c>.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the backend cannot be initialized due to missing dependencies
    /// or system configuration issues.
    /// </exception>
    Task InitializeAsync(CancellationToken ct = default);

    /// <summary>
    /// Enumerates all available compute devices supported by this backend.
    /// </summary>
    /// <param name="ct">A cancellation token to observe while waiting for the task to complete.</param>
    /// <returns>
    /// A task that represents the asynchronous enumeration operation. The result contains
    /// a read-only list of discovered compute devices.
    /// </returns>
    /// <remarks>
    /// This method discovers and returns all compute devices that are available through
    /// this backend. The enumeration should include device capabilities, memory information,
    /// and availability status. This method should only be called after successful
    /// initialization via <see cref="InitializeAsync"/>.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when device enumeration is attempted before backend initialization
    /// or when the backend is in an invalid state.
    /// </exception>
    Task<IReadOnlyList<IComputeDevice>> EnumerateDevicesAsync(CancellationToken ct = default);
}