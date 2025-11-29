using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Diagnostics.Interfaces;
using Orleans.GpuBridge.Diagnostics.Enums;

namespace Orleans.GpuBridge.Diagnostics.Implementation;

/// <summary>
/// Implementation of GPU telemetry collection using OpenTelemetry metrics and distributed tracing.
/// This service provides comprehensive monitoring of GPU operations, performance metrics, and resource utilization
/// through industry-standard observability patterns and tools.
/// </summary>
/// <remarks>
/// This implementation uses:
/// - OpenTelemetry Metrics API for counters, histograms, and observable gauges
/// - OpenTelemetry Tracing API for distributed tracing spans and activities  
/// - Structured logging for diagnostic information
/// - Thread-safe observable gauge callbacks for real-time metrics
/// </remarks>
public sealed class GpuTelemetry : IGpuTelemetry, IDisposable
{
    private readonly ILogger<GpuTelemetry> _logger;
    private readonly Meter _meter;
    private readonly ActivitySource _activitySource;

    #region Metric Instruments - Counters

    /// <summary>Counter tracking the total number of GPU kernels successfully executed.</summary>
    private readonly Counter<long> _kernelsExecuted;

    /// <summary>Counter tracking the total number of GPU kernel execution failures.</summary>
    private readonly Counter<long> _kernelFailures;

    /// <summary>Counter tracking the total bytes transferred to/from GPU memory.</summary>
    private readonly Counter<long> _bytesTransferred;

    /// <summary>Counter tracking successful GPU memory allocations.</summary>
    private readonly Counter<long> _allocationSuccesses;

    /// <summary>Counter tracking failed GPU memory allocations.</summary>
    private readonly Counter<long> _allocationFailures;

    /// <summary>Counter tracking GPU-related grain activations.</summary>
    private readonly Counter<long> _grainActivations;

    /// <summary>Counter tracking pipeline stage executions.</summary>
    private readonly Counter<long> _pipelineExecutions;

    #endregion

    #region Metric Instruments - Histograms

    /// <summary>Histogram tracking GPU kernel execution latency distribution.</summary>
    private readonly Histogram<double> _kernelLatency;

    /// <summary>Histogram tracking GPU memory transfer throughput distribution.</summary>
    private readonly Histogram<double> _transferThroughput;

    /// <summary>Histogram tracking GPU memory allocation size distribution.</summary>
    private readonly Histogram<double> _allocationSize;

    /// <summary>Histogram tracking GPU grain activation latency distribution.</summary>
    private readonly Histogram<double> _grainActivationLatency;

    /// <summary>Histogram tracking pipeline stage execution latency distribution.</summary>
    private readonly Histogram<double> _pipelineStageLatency;

    #endregion

    #region Metric Instruments - Observable Gauges

    /// <summary>Observable gauge reporting current GPU memory usage per device.</summary>
    private readonly ObservableGauge<long> _gpuMemoryUsed;

    /// <summary>Observable gauge reporting current GPU utilization percentage per device.</summary>
    private readonly ObservableGauge<double> _gpuUtilization;

    /// <summary>Observable gauge reporting current work queue depth per device.</summary>
    private readonly ObservableGauge<int> _queueDepth;

    /// <summary>Observable gauge reporting current GPU temperature per device.</summary>
    private readonly ObservableGauge<double> _gpuTemperature;

    /// <summary>Observable gauge reporting current GPU power consumption per device.</summary>
    private readonly ObservableGauge<double> _gpuPowerUsage;

    #endregion

    #region State for Observable Gauges (Thread-Safe)

    /// <summary>Thread-safe dictionary tracking memory usage by device index.</summary>
    private readonly Dictionary<int, long> _memoryUsage = new();

    /// <summary>Thread-safe dictionary tracking GPU utilization by device index.</summary>
    private readonly Dictionary<int, double> _utilization = new();

    /// <summary>Thread-safe dictionary tracking work queue depths by device index.</summary>
    private readonly Dictionary<int, int> _queueDepths = new();

    /// <summary>Thread-safe dictionary tracking GPU temperatures by device index.</summary>
    private readonly Dictionary<int, double> _temperatures = new();

    /// <summary>Thread-safe dictionary tracking GPU power usage by device index.</summary>
    private readonly Dictionary<int, double> _powerUsage = new();

    #endregion

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuTelemetry"/> class with OpenTelemetry integration.
    /// </summary>
    /// <param name="logger">Logger for recording telemetry operations and diagnostics.</param>
    /// <param name="meterFactory">Factory for creating OpenTelemetry meters and metric instruments.</param>
    public GpuTelemetry(ILogger<GpuTelemetry> logger, IMeterFactory meterFactory)
    {
        _logger = logger;
        _meter = meterFactory.Create("Orleans.GpuBridge", "1.0.0");
        _activitySource = new ActivitySource("Orleans.GpuBridge", "1.0.0");

        // Initialize counter instruments
        _kernelsExecuted = _meter.CreateCounter<long>(
            "gpu.kernels.executed",
            unit: "kernels",
            description: "Number of GPU kernels executed successfully");

        _kernelFailures = _meter.CreateCounter<long>(
            "gpu.kernels.failed",
            unit: "kernels",
            description: "Number of GPU kernel execution failures");

        _bytesTransferred = _meter.CreateCounter<long>(
            "gpu.bytes.transferred",
            unit: "bytes",
            description: "Total bytes transferred between host and GPU memory");

        _allocationSuccesses = _meter.CreateCounter<long>(
            "gpu.allocations.succeeded",
            unit: "allocations",
            description: "Number of successful GPU memory allocations");

        _allocationFailures = _meter.CreateCounter<long>(
            "gpu.allocations.failed",
            unit: "allocations",
            description: "Number of failed GPU memory allocations");

        _grainActivations = _meter.CreateCounter<long>(
            "gpu.grains.activated",
            unit: "grains",
            description: "Number of GPU-related grain activations");

        _pipelineExecutions = _meter.CreateCounter<long>(
            "gpu.pipeline.executions",
            unit: "executions",
            description: "Number of GPU pipeline stage executions");

        // Initialize histogram instruments
        _kernelLatency = _meter.CreateHistogram<double>(
            "gpu.kernel.latency",
            unit: "milliseconds",
            description: "GPU kernel execution latency distribution");

        _transferThroughput = _meter.CreateHistogram<double>(
            "gpu.transfer.throughput",
            unit: "GB/s",
            description: "GPU memory transfer throughput distribution");

        _allocationSize = _meter.CreateHistogram<double>(
            "gpu.allocation.size",
            unit: "bytes",
            description: "GPU memory allocation size distribution");

        _grainActivationLatency = _meter.CreateHistogram<double>(
            "gpu.grain.activation.latency",
            unit: "milliseconds",
            description: "GPU grain activation latency distribution");

        _pipelineStageLatency = _meter.CreateHistogram<double>(
            "gpu.pipeline.stage.latency",
            unit: "milliseconds",
            description: "GPU pipeline stage execution latency distribution");

        // Initialize observable gauge instruments
        _gpuMemoryUsed = _meter.CreateObservableGauge<long>(
            "gpu.memory.used",
            GetGpuMemoryUsed,
            unit: "bytes",
            description: "GPU memory currently in use per device");

        _gpuUtilization = _meter.CreateObservableGauge<double>(
            "gpu.utilization",
            GetGpuUtilization,
            unit: "percent",
            description: "GPU compute utilization percentage per device");

        _queueDepth = _meter.CreateObservableGauge<int>(
            "gpu.queue.depth",
            GetQueueDepth,
            unit: "items",
            description: "Number of items in GPU work queue per device");

        _gpuTemperature = _meter.CreateObservableGauge<double>(
            "gpu.temperature",
            GetGpuTemperature,
            unit: "celsius",
            description: "GPU temperature in Celsius per device");

        _gpuPowerUsage = _meter.CreateObservableGauge<double>(
            "gpu.power.usage",
            GetGpuPowerUsage,
            unit: "watts",
            description: "GPU power consumption in watts per device");

        _logger.LogInformation("GPU telemetry service initialized with OpenTelemetry metrics and distributed tracing support");
    }

    /// <inheritdoc />
    public Activity? StartKernelExecution(string kernelName, int deviceIndex)
    {
        var activity = _activitySource.StartActivity(
            "gpu.kernel.execute",
            ActivityKind.Internal);

        if (activity != null)
        {
            // Set standard OpenTelemetry semantic attributes
            activity.SetTag("kernel.name", kernelName);
            activity.SetTag("device.index", deviceIndex);
            activity.SetTag("device.type", "gpu");
            activity.SetTag("execution.type", "kernel");

            // Add correlation baggage for distributed tracing
            activity.SetBaggage("kernel.id", $"{kernelName}_{deviceIndex}");
        }

        return activity;
    }

    /// <inheritdoc />
    public void RecordKernelExecution(string kernelName, int deviceIndex, TimeSpan duration, bool success)
    {
        var tags = new TagList
        {
            { "kernel", kernelName },
            { "device", deviceIndex },
            { "success", success }
        };

        // Update appropriate counters
        if (success)
        {
            _kernelsExecuted.Add(1, tags);
        }
        else
        {
            _kernelFailures.Add(1, tags);
        }

        // Record execution latency
        _kernelLatency.Record(duration.TotalMilliseconds, tags);

        // Update current activity if present
        var activity = Activity.Current;
        if (activity != null)
        {
            activity.SetTag("execution.duration_ms", duration.TotalMilliseconds);
            activity.SetTag("execution.success", success);

            if (!success)
            {
                activity.SetStatus(ActivityStatusCode.Error, "Kernel execution failed");
            }
        }

        _logger.LogDebug(
            "Kernel {KernelName} on device {DeviceIndex} executed in {Duration}ms (Success: {Success})",
            kernelName, deviceIndex, duration.TotalMilliseconds, success);
    }

    /// <inheritdoc />
    public void RecordMemoryTransfer(TransferDirection direction, long bytes, TimeSpan duration)
    {
        var tags = new TagList
        {
            { "direction", direction.ToString() }
        };

        _bytesTransferred.Add(bytes, tags);

        // Calculate and record throughput in GB/s
        var throughputGbps = (bytes / 1e9) / duration.TotalSeconds;
        _transferThroughput.Record(throughputGbps, tags);

        // Create distributed tracing span for transfer operation
        using var activity = _activitySource.StartActivity(
            "gpu.memory.transfer",
            ActivityKind.Internal);

        if (activity != null)
        {
            activity.SetTag("transfer.direction", direction.ToString());
            activity.SetTag("transfer.bytes", bytes);
            activity.SetTag("transfer.throughput_gbps", throughputGbps);
            activity.SetTag("transfer.duration_ms", duration.TotalMilliseconds);
        }
    }

    /// <inheritdoc />
    public void RecordMemoryAllocation(int deviceIndex, long bytes, bool success)
    {
        var tags = new TagList
        {
            { "device", deviceIndex },
            { "success", success }
        };

        if (success)
        {
            _allocationSuccesses.Add(1, tags);
            _allocationSize.Record(bytes, tags);

            // Update memory usage tracking for observable gauge
            lock (_memoryUsage)
            {
                if (!_memoryUsage.ContainsKey(deviceIndex))
                    _memoryUsage[deviceIndex] = 0;
                _memoryUsage[deviceIndex] += bytes;
            }
        }
        else
        {
            _allocationFailures.Add(1, tags);
        }
    }

    /// <inheritdoc />
    public void RecordAllocationFailure(int deviceIndex, long requestedBytes, string reason)
    {
        var tags = new TagList
        {
            { "device", deviceIndex },
            { "reason", reason }
        };

        _allocationFailures.Add(1, tags);

        // Update current activity if present
        var activity = Activity.Current;
        if (activity != null)
        {
            activity.SetStatus(
                ActivityStatusCode.Error,
                $"GPU allocation failed: {reason}");
            activity.SetTag("allocation.requested_bytes", requestedBytes);
            activity.SetTag("allocation.failure_reason", reason);
        }

        _logger.LogWarning(
            "GPU memory allocation failed on device {DeviceIndex}: requested {Bytes} bytes, reason: {Reason}",
            deviceIndex, requestedBytes, reason);
    }

    /// <inheritdoc />
    public void RecordQueueDepth(int deviceIndex, int depth)
    {
        lock (_queueDepths)
        {
            _queueDepths[deviceIndex] = depth;
        }
    }

    /// <inheritdoc />
    public void RecordGrainActivation(string grainType, TimeSpan duration)
    {
        var tags = new TagList
        {
            { "grain_type", grainType }
        };

        _grainActivations.Add(1, tags);
        _grainActivationLatency.Record(duration.TotalMilliseconds, tags);
    }

    /// <inheritdoc />
    public void RecordPipelineStage(string stageName, TimeSpan duration, bool success)
    {
        var tags = new TagList
        {
            { "stage", stageName },
            { "success", success }
        };

        _pipelineExecutions.Add(1, tags);
        _pipelineStageLatency.Record(duration.TotalMilliseconds, tags);
    }

    #region Observable Gauge Callback Methods

    /// <summary>
    /// Callback method for the GPU memory usage observable gauge.
    /// Returns current memory usage for all tracked devices.
    /// </summary>
    private IEnumerable<Measurement<long>> GetGpuMemoryUsed()
    {
        lock (_memoryUsage)
        {
            foreach (var (device, usage) in _memoryUsage)
            {
                yield return new Measurement<long>(
                    usage,
                    new TagList { { "device", device } });
            }
        }
    }

    /// <summary>
    /// Callback method for the GPU utilization observable gauge.
    /// Returns current utilization percentage for all tracked devices.
    /// </summary>
    private IEnumerable<Measurement<double>> GetGpuUtilization()
    {
        lock (_utilization)
        {
            foreach (var (device, util) in _utilization)
            {
                yield return new Measurement<double>(
                    util,
                    new TagList { { "device", device } });
            }
        }
    }

    /// <summary>
    /// Callback method for the queue depth observable gauge.
    /// Returns current work queue depth for all tracked devices.
    /// </summary>
    private IEnumerable<Measurement<int>> GetQueueDepth()
    {
        lock (_queueDepths)
        {
            foreach (var (device, depth) in _queueDepths)
            {
                yield return new Measurement<int>(
                    depth,
                    new TagList { { "device", device } });
            }
        }
    }

    /// <summary>
    /// Callback method for the GPU temperature observable gauge.
    /// Returns current temperature for all tracked devices.
    /// </summary>
    private IEnumerable<Measurement<double>> GetGpuTemperature()
    {
        lock (_temperatures)
        {
            foreach (var (device, temp) in _temperatures)
            {
                yield return new Measurement<double>(
                    temp,
                    new TagList { { "device", device } });
            }
        }
    }

    /// <summary>
    /// Callback method for the GPU power usage observable gauge.
    /// Returns current power consumption for all tracked devices.
    /// </summary>
    private IEnumerable<Measurement<double>> GetGpuPowerUsage()
    {
        lock (_powerUsage)
        {
            foreach (var (device, power) in _powerUsage)
            {
                yield return new Measurement<double>(
                    power,
                    new TagList { { "device", device } });
            }
        }
    }

    #endregion

    /// <summary>
    /// Updates GPU hardware metrics from external monitoring systems.
    /// This method should be called periodically by GPU monitoring services
    /// to provide real-time hardware statistics for the observable gauges.
    /// </summary>
    /// <param name="deviceIndex">The index of the GPU device.</param>
    /// <param name="utilization">Current GPU utilization percentage (0.0 - 100.0).</param>
    /// <param name="temperature">Current GPU temperature in Celsius.</param>
    /// <param name="power">Current GPU power consumption in watts.</param>
    public void UpdateGpuMetrics(int deviceIndex, double utilization, double temperature, double power)
    {
        lock (_utilization)
        {
            _utilization[deviceIndex] = utilization;
        }

        lock (_temperatures)
        {
            _temperatures[deviceIndex] = temperature;
        }

        lock (_powerUsage)
        {
            _powerUsage[deviceIndex] = power;
        }
    }

    /// <summary>
    /// Releases all resources used by the GPU telemetry service.
    /// This properly disposes of OpenTelemetry meters and activity sources.
    /// </summary>
    public void Dispose()
    {
        _meter?.Dispose();
        _activitySource?.Dispose();
    }
}