using System;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Abstractions.Models;
using Orleans.GpuBridge.Abstractions.Providers.Memory.Interfaces;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Parameters;
using Orleans.GpuBridge.Backends.DotCompute.Memory;
using DotComputeKernelArguments = DotCompute.Abstractions.Kernels.KernelArguments;

namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Helper methods for DotComputeKernelExecutor
/// </summary>
internal sealed partial class DotComputeKernelExecutor
{
    /// <summary>
    /// Selects the optimal device for kernel execution
    /// </summary>
    private IComputeDevice SelectExecutionDevice(KernelExecutionParameters parameters)
    {
        // Use preferred device from parameters if available
        if (parameters.PreferredQueue?.Context?.Device != null)
        {
            return parameters.PreferredQueue.Context.Device;
        }

        // Select based on device requirements (simplified)
        var devices = _deviceManager.GetDevices();
        var gpuDevices = devices.Where(d => d.Type != DeviceType.CPU).ToList();

        return gpuDevices.FirstOrDefault() ?? devices.FirstOrDefault() ??
               throw new InvalidOperationException("No suitable device available for kernel execution");
    }

    /// <summary>
    /// Prepares DotCompute kernel arguments from Orleans execution parameters
    /// </summary>
    /// <remarks>
    /// Phase 1.3: Updated to use native IUnifiedMemoryBuffer directly
    ///
    /// Converts Orleans.GpuBridge memory and scalar arguments to DotCompute KernelArguments.
    ///
    /// For memory arguments:
    /// - Uses native IUnifiedMemoryBuffer from DotComputeDeviceMemoryWrapper
    /// - Zero-copy execution - no temporary buffer allocation
    ///
    /// For scalar arguments:
    /// - Directly passes through using KernelArguments.AddScalar()
    /// </remarks>
    private Task<DotComputeKernelArguments> PrepareKernelArgumentsAsync(
        KernelExecutionParameters parameters,
        IComputeDevice device,
        CancellationToken cancellationToken)
    {
        // Create kernel arguments with capacity hint
        var totalArgs = parameters.MemoryArguments.Count + parameters.ScalarArguments.Count;
        var kernelArgs = new DotComputeKernelArguments(totalArgs);

        // Process memory arguments
        foreach (var memArg in parameters.MemoryArguments)
        {
            if (memArg.Value is DotComputeDeviceMemoryWrapper dotComputeMemory)
            {
                // ✅ Phase 1.3: Use native buffer directly (zero-copy)
                if (dotComputeMemory.NativeBuffer != null)
                {
                    kernelArgs.AddBuffer(dotComputeMemory.NativeBuffer);

                    _logger.LogDebug(
                        "Added native buffer for argument '{ArgName}' ({SizeBytes} bytes)",
                        memArg.Key,
                        dotComputeMemory.SizeBytes);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Memory argument '{memArg.Key}' does not have a native buffer. " +
                        "This may be due to legacy allocation. Please recreate the memory buffer.");
                }
            }
            else if (memArg.Value is IDeviceMemory deviceMemory)
            {
                throw new InvalidOperationException(
                    $"Memory argument '{memArg.Key}' is not a DotComputeDeviceMemoryWrapper. " +
                    $"Cannot use memory from other backends. Got type: {deviceMemory.GetType().Name}");
            }
            else
            {
                throw new InvalidOperationException(
                    $"Memory argument '{memArg.Key}' is not an IDeviceMemory instance");
            }
        }

        // Process scalar arguments - these work directly
        foreach (var scalarArg in parameters.ScalarArguments)
        {
            kernelArgs.AddScalar(scalarArg.Value);
        }

        _logger.LogDebug(
            "Prepared DotCompute kernel arguments: {BufferCount} buffers, {ScalarCount} scalars (zero-copy)",
            parameters.MemoryArguments.Count,
            parameters.ScalarArguments.Count);

        return Task.FromResult(kernelArgs);
    }

    /// <summary>
    /// Calculates work dimensions from kernel execution parameters
    /// </summary>
    private WorkDimensions CalculateWorkDimensions(KernelExecutionParameters parameters)
    {
        var globalSize = parameters.GlobalWorkSize.Length > 0 ? parameters.GlobalWorkSize : new[] { 1 };
        var localSize = parameters.LocalWorkSize?.Length > 0 ? parameters.LocalWorkSize : null;

        return new WorkDimensions(globalSize, localSize);
    }
}
