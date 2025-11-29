using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using DotCompute.Abstractions.Health;
using DotCompute.Abstractions.Kernels;
using DotCompute.Abstractions.Kernels.Types;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Abstractions.Kernels;
using Orleans.GpuBridge.Abstractions.Providers;

namespace Orleans.GpuBridge.Backends.DotCompute;

/// <summary>
/// DotCompute accelerator provider for Orleans.GpuBridge.
/// Manages IAccelerator instances and kernel lifecycle.
/// </summary>
public sealed class DotComputeAcceleratorProvider : IDisposable
{
    private readonly ConcurrentDictionary<string, IAccelerator> _accelerators = new();
    private readonly ConcurrentDictionary<string, object> _kernels = new();
    private readonly CompilationOptions _defaultCompilationOptions;
    private bool _isDisposed;

    /// <summary>
    /// Gets all available accelerators.
    /// </summary>
    public IReadOnlyCollection<IAccelerator> Accelerators => _accelerators.Values.ToList().AsReadOnly();

    /// <summary>
    /// Gets the primary GPU accelerator (if available).
    /// </summary>
    public IAccelerator? PrimaryGpuAccelerator =>
        _accelerators.Values.FirstOrDefault(a =>
            a.Type != AcceleratorType.CPU && a.IsAvailable);

    /// <summary>
    /// Gets the CPU accelerator (fallback).
    /// </summary>
    public IAccelerator? CpuAccelerator =>
        _accelerators.Values.FirstOrDefault(a => a.Type == AcceleratorType.CPU);

    /// <summary>
    /// Creates a new DotComputeAcceleratorProvider.
    /// </summary>
    /// <param name="compilationOptions">Default compilation options (null = default)</param>
    public DotComputeAcceleratorProvider(CompilationOptions? compilationOptions = null)
    {
        _defaultCompilationOptions = compilationOptions ?? CompilationOptions.Default;
    }

    /// <summary>
    /// Registers an accelerator with the provider.
    /// </summary>
    /// <param name="accelerator">Accelerator to register</param>
    /// <param name="name">Optional name (defaults to accelerator.Info.Name)</param>
    public void RegisterAccelerator(IAccelerator accelerator, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);

        var acceleratorName = name ?? accelerator.Info.Name;

        if (!_accelerators.TryAdd(acceleratorName, accelerator))
        {
            throw new InvalidOperationException(
                $"Accelerator '{acceleratorName}' is already registered.");
        }
    }

    /// <summary>
    /// Gets an accelerator by name.
    /// </summary>
    /// <param name="name">Accelerator name</param>
    /// <returns>Accelerator instance</returns>
    public IAccelerator? GetAccelerator(string name)
    {
        _accelerators.TryGetValue(name, out var accelerator);
        return accelerator;
    }

    /// <summary>
    /// Creates a DotCompute kernel for Orleans.GpuBridge.
    /// </summary>
    /// <typeparam name="TIn">Input type</typeparam>
    /// <typeparam name="TOut">Output type</typeparam>
    /// <param name="kernelName">Unique kernel name</param>
    /// <param name="kernelSource">Kernel source code</param>
    /// <param name="language">Kernel language (defaults to CSharp)</param>
    /// <param name="entryPoint">Entry point function name (defaults to "main")</param>
    /// <param name="compilationOptions">Compilation options (null = default)</param>
    /// <param name="preferGpu">Prefer GPU accelerator over CPU (default: true)</param>
    /// <param name="inputConverter">Custom input converter (null = use default)</param>
    /// <param name="outputConverter">Custom output converter (null = use default)</param>
    /// <returns>Configured DotComputeKernel instance</returns>
    public DotComputeKernel<TIn, TOut> CreateKernel<TIn, TOut>(
        string kernelName,
        string kernelSource,
        KernelLanguage language = KernelLanguage.CSharp,
        string entryPoint = "main",
        CompilationOptions? compilationOptions = null,
        bool preferGpu = true,
        Func<TIn, KernelArgument[]>? inputConverter = null,
        Func<IUnifiedMemoryBuffer, TOut>? outputConverter = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelName);
        ArgumentException.ThrowIfNullOrWhiteSpace(kernelSource);

        // Select accelerator (GPU preferred, CPU fallback)
        var accelerator = SelectAccelerator(preferGpu);
        if (accelerator == null)
        {
            throw new InvalidOperationException(
                "No accelerators available. Register at least one accelerator.");
        }

        // Create kernel definition
        var kernelDefinition = new KernelDefinition(kernelName, kernelSource, entryPoint)
        {
            Language = language
        };

        // Create kernel instance
        var kernel = new DotComputeKernel<TIn, TOut>(
            accelerator,
            kernelDefinition,
            compilationOptions ?? _defaultCompilationOptions,
            inputConverter,
            outputConverter);

        // Cache kernel for reuse
        var cacheKey = $"{kernelName}_{typeof(TIn).Name}_{typeof(TOut).Name}";
        _kernels.TryAdd(cacheKey, kernel);

        return kernel;
    }

    /// <summary>
    /// Creates a DotCompute kernel from an existing KernelDefinition.
    /// </summary>
    /// <typeparam name="TIn">Input type</typeparam>
    /// <typeparam name="TOut">Output type</typeparam>
    /// <param name="kernelDefinition">Kernel definition</param>
    /// <param name="compilationOptions">Compilation options (null = default)</param>
    /// <param name="preferGpu">Prefer GPU accelerator over CPU (default: true)</param>
    /// <param name="inputConverter">Custom input converter (null = use default)</param>
    /// <param name="outputConverter">Custom output converter (null = use default)</param>
    /// <returns>Configured DotComputeKernel instance</returns>
    public DotComputeKernel<TIn, TOut> CreateKernel<TIn, TOut>(
        KernelDefinition kernelDefinition,
        CompilationOptions? compilationOptions = null,
        bool preferGpu = true,
        Func<TIn, KernelArgument[]>? inputConverter = null,
        Func<IUnifiedMemoryBuffer, TOut>? outputConverter = null)
    {
        ArgumentNullException.ThrowIfNull(kernelDefinition);

        var accelerator = SelectAccelerator(preferGpu);
        if (accelerator == null)
        {
            throw new InvalidOperationException(
                "No accelerators available. Register at least one accelerator.");
        }

        return new DotComputeKernel<TIn, TOut>(
            accelerator,
            kernelDefinition,
            compilationOptions ?? _defaultCompilationOptions,
            inputConverter,
            outputConverter);
    }

    /// <summary>
    /// Gets GPU device information for all registered accelerators.
    /// </summary>
    /// <returns>Collection of GPU devices</returns>
    public async Task<IReadOnlyList<GpuDevice>> GetDevicesAsync(CancellationToken cancellationToken = default)
    {
        var devices = new List<GpuDevice>();

        foreach (var accelerator in _accelerators.Values)
        {
            var info = accelerator.Info;

            // Get health snapshot for memory info
            DeviceHealthSnapshot? health = null;
            try
            {
                health = await accelerator.GetHealthSnapshotAsync(cancellationToken);
            }
            catch
            {
                // Best-effort health snapshot
            }

            // Map DotCompute AcceleratorType to Orleans.GpuBridge DeviceType
            var deviceType = accelerator.Type switch
            {
                AcceleratorType.CPU => Abstractions.Enums.DeviceType.CPU,
                AcceleratorType.CUDA => Abstractions.Enums.DeviceType.CUDA,
                AcceleratorType.OpenCL => Abstractions.Enums.DeviceType.OpenCL,
                AcceleratorType.Metal => Abstractions.Enums.DeviceType.Metal,
                _ => Abstractions.Enums.DeviceType.CPU
            };

            // Extract memory from sensor readings if available
            var memoryTotalSensor = health?.GetSensorValue(SensorType.MemoryTotalBytes);
            var memoryUsedSensor = health?.GetSensorValue(SensorType.MemoryUsedBytes);

            var totalMemory = memoryTotalSensor.HasValue ? (long)memoryTotalSensor.Value : 0;
            var usedMemory = memoryUsedSensor.HasValue ? (long)memoryUsedSensor.Value : 0;
            var availableMemory = totalMemory - usedMemory;

            var capabilities = new List<string>
            {
                $"Backend: DotCompute",
                $"Type: {accelerator.Type}",
                $"Available: {accelerator.IsAvailable}"
            };

            var device = new GpuDevice(
                Index: devices.Count,
                Name: info.Name,
                Type: deviceType,
                TotalMemoryBytes: totalMemory,
                AvailableMemoryBytes: availableMemory,
                ComputeUnits: info.MaxComputeUnits,
                Capabilities: capabilities.AsReadOnly()
            );

            devices.Add(device);
        }

        return devices.AsReadOnly();
    }

    /// <summary>
    /// Gets GpuBridge information.
    /// </summary>
    /// <returns>GpuBridge info</returns>
    public GpuBridgeInfo GetBridgeInfo()
    {
        var totalMemory = _accelerators.Values
            .Sum(a =>
            {
                try
                {
                    var health = a.GetHealthSnapshotAsync().AsTask().Result;
                    var memorySensor = health?.GetSensorValue(SensorType.MemoryTotalBytes);
                    return memorySensor.HasValue ? (long)memorySensor.Value : 0L;
                }
                catch
                {
                    return 0L;
                }
            });

        // Map backend type
        var backend = PrimaryGpuAccelerator?.Type switch
        {
            AcceleratorType.CUDA => Abstractions.Enums.GpuBackend.CUDA,
            AcceleratorType.OpenCL => Abstractions.Enums.GpuBackend.OpenCL,
            AcceleratorType.Metal => Abstractions.Enums.GpuBackend.Metal,
            _ => Abstractions.Enums.GpuBackend.CPU
        };

        var metadata = new Dictionary<string, object>
        {
            ["Backend"] = "DotCompute",
            ["AcceleratorCount"] = _accelerators.Count,
            ["HasGpu"] = PrimaryGpuAccelerator != null
        };

        return new GpuBridgeInfo(
            Version: "1.0.0-dotcompute",
            DeviceCount: _accelerators.Count,
            TotalMemoryBytes: totalMemory,
            Backend: backend,
            IsGpuAvailable: PrimaryGpuAccelerator?.IsAvailable ?? false,
            Metadata: metadata
        );
    }

    /// <summary>
    /// Synchronizes all accelerators (waits for GPU completion).
    /// </summary>
    public async Task SynchronizeAllAsync(CancellationToken cancellationToken = default)
    {
        var syncTasks = _accelerators.Values
            .Select(a => a.SynchronizeAsync(cancellationToken).AsTask());

        await Task.WhenAll(syncTasks);
    }

    /// <summary>
    /// Disposes all accelerators and cleans up resources.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        // Dispose all kernels
        foreach (var kernel in _kernels.Values)
        {
            if (kernel is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
        _kernels.Clear();

        // Dispose all accelerators
        foreach (var accelerator in _accelerators.Values)
        {
            accelerator.DisposeAsync().AsTask().Wait();
        }
        _accelerators.Clear();

        _isDisposed = true;
    }

    /// <summary>
    /// Selects the best accelerator based on preference.
    /// </summary>
    /// <param name="preferGpu">Prefer GPU over CPU</param>
    /// <returns>Selected accelerator</returns>
    private IAccelerator? SelectAccelerator(bool preferGpu)
    {
        if (preferGpu)
        {
            // Try GPU first, fallback to CPU
            return PrimaryGpuAccelerator ?? CpuAccelerator;
        }

        // Try CPU first, fallback to GPU
        return CpuAccelerator ?? PrimaryGpuAccelerator;
    }
}
