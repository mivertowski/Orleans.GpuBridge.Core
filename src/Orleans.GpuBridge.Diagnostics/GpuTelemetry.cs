using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Diagnostics;

public interface IGpuTelemetry
{
    Activity? StartKernelExecution(string kernelName, int deviceIndex);
    void RecordKernelExecution(string kernelName, int deviceIndex, TimeSpan duration, bool success);
    void RecordMemoryTransfer(TransferDirection direction, long bytes, TimeSpan duration);
    void RecordMemoryAllocation(int deviceIndex, long bytes, bool success);
    void RecordAllocationFailure(int deviceIndex, long requestedBytes, string reason);
    void RecordQueueDepth(int deviceIndex, int depth);
    void RecordGrainActivation(string grainType, TimeSpan duration);
    void RecordPipelineStage(string stageName, TimeSpan duration, bool success);
}

public sealed class GpuTelemetry : IGpuTelemetry, IDisposable
{
    private readonly ILogger<GpuTelemetry> _logger;
    private readonly Meter _meter;
    private readonly ActivitySource _activitySource;
    
    // Metrics - Counters
    private readonly Counter<long> _kernelsExecuted;
    private readonly Counter<long> _kernelFailures;
    private readonly Counter<long> _bytesTransferred;
    private readonly Counter<long> _allocationSuccesses;
    private readonly Counter<long> _allocationFailures;
    private readonly Counter<long> _grainActivations;
    private readonly Counter<long> _pipelineExecutions;
    
    // Metrics - Histograms
    private readonly Histogram<double> _kernelLatency;
    private readonly Histogram<double> _transferThroughput;
    private readonly Histogram<double> _allocationSize;
    private readonly Histogram<double> _grainActivationLatency;
    private readonly Histogram<double> _pipelineStageLatency;
    
    // Metrics - Observable Gauges
    private readonly ObservableGauge<long> _gpuMemoryUsed;
    private readonly ObservableGauge<double> _gpuUtilization;
    private readonly ObservableGauge<int> _queueDepth;
    private readonly ObservableGauge<double> _gpuTemperature;
    private readonly ObservableGauge<double> _gpuPowerUsage;
    
    // State for observable gauges
    private readonly Dictionary<int, long> _memoryUsage = new();
    private readonly Dictionary<int, double> _utilization = new();
    private readonly Dictionary<int, int> _queueDepths = new();
    private readonly Dictionary<int, double> _temperatures = new();
    private readonly Dictionary<int, double> _powerUsage = new();
    
    public GpuTelemetry(ILogger<GpuTelemetry> logger, IMeterFactory meterFactory)
    {
        _logger = logger;
        _meter = meterFactory.Create("Orleans.GpuBridge", "1.0.0");
        _activitySource = new ActivitySource("Orleans.GpuBridge", "1.0.0");
        
        // Initialize counters
        _kernelsExecuted = _meter.CreateCounter<long>(
            "gpu.kernels.executed",
            unit: "kernels",
            description: "Number of GPU kernels executed");
        
        _kernelFailures = _meter.CreateCounter<long>(
            "gpu.kernels.failed",
            unit: "kernels",
            description: "Number of GPU kernel execution failures");
        
        _bytesTransferred = _meter.CreateCounter<long>(
            "gpu.bytes.transferred",
            unit: "bytes",
            description: "Total bytes transferred to/from GPU");
        
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
            description: "Number of GPU grain activations");
        
        _pipelineExecutions = _meter.CreateCounter<long>(
            "gpu.pipeline.executions",
            unit: "executions",
            description: "Number of pipeline executions");
        
        // Initialize histograms
        _kernelLatency = _meter.CreateHistogram<double>(
            "gpu.kernel.latency",
            unit: "milliseconds",
            description: "GPU kernel execution latency");
        
        _transferThroughput = _meter.CreateHistogram<double>(
            "gpu.transfer.throughput",
            unit: "GB/s",
            description: "GPU memory transfer throughput");
        
        _allocationSize = _meter.CreateHistogram<double>(
            "gpu.allocation.size",
            unit: "bytes",
            description: "GPU memory allocation sizes");
        
        _grainActivationLatency = _meter.CreateHistogram<double>(
            "gpu.grain.activation.latency",
            unit: "milliseconds",
            description: "GPU grain activation latency");
        
        _pipelineStageLatency = _meter.CreateHistogram<double>(
            "gpu.pipeline.stage.latency",
            unit: "milliseconds",
            description: "Pipeline stage execution latency");
        
        // Initialize observable gauges
        _gpuMemoryUsed = _meter.CreateObservableGauge<long>(
            "gpu.memory.used",
            GetGpuMemoryUsed,
            unit: "bytes",
            description: "GPU memory currently in use");
        
        _gpuUtilization = _meter.CreateObservableGauge<double>(
            "gpu.utilization",
            GetGpuUtilization,
            unit: "percent",
            description: "GPU compute utilization percentage");
        
        _queueDepth = _meter.CreateObservableGauge<int>(
            "gpu.queue.depth",
            GetQueueDepth,
            unit: "items",
            description: "Number of items in GPU work queue");
        
        _gpuTemperature = _meter.CreateObservableGauge<double>(
            "gpu.temperature",
            GetGpuTemperature,
            unit: "celsius",
            description: "GPU temperature in Celsius");
        
        _gpuPowerUsage = _meter.CreateObservableGauge<double>(
            "gpu.power.usage",
            GetGpuPowerUsage,
            unit: "watts",
            description: "GPU power consumption in watts");
        
        _logger.LogInformation("GPU telemetry initialized with OpenTelemetry metrics and tracing");
    }
    
    public Activity? StartKernelExecution(string kernelName, int deviceIndex)
    {
        var activity = _activitySource.StartActivity(
            "gpu.kernel.execute",
            ActivityKind.Internal);
        
        if (activity != null)
        {
            activity.SetTag("kernel.name", kernelName);
            activity.SetTag("device.index", deviceIndex);
            activity.SetTag("device.type", "gpu");
            activity.SetTag("execution.type", "kernel");
            
            // Add baggage for correlation
            activity.SetBaggage("kernel.id", $"{kernelName}_{deviceIndex}");
        }
        
        return activity;
    }
    
    public void RecordKernelExecution(string kernelName, int deviceIndex, TimeSpan duration, bool success)
    {
        var tags = new TagList
        {
            { "kernel", kernelName },
            { "device", deviceIndex },
            { "success", success }
        };
        
        if (success)
        {
            _kernelsExecuted.Add(1, tags);
        }
        else
        {
            _kernelFailures.Add(1, tags);
        }
        
        _kernelLatency.Record(duration.TotalMilliseconds, tags);
        
        // Update activity if present
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
    
    public void RecordMemoryTransfer(TransferDirection direction, long bytes, TimeSpan duration)
    {
        var tags = new TagList
        {
            { "direction", direction.ToString() }
        };
        
        _bytesTransferred.Add(bytes, tags);
        
        // Calculate throughput in GB/s
        var throughputGbps = (bytes / 1e9) / duration.TotalSeconds;
        _transferThroughput.Record(throughputGbps, tags);
        
        // Create span for transfer
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
            
            // Update memory usage
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
    
    public void RecordAllocationFailure(int deviceIndex, long requestedBytes, string reason)
    {
        var tags = new TagList
        {
            { "device", deviceIndex },
            { "reason", reason }
        };
        
        _allocationFailures.Add(1, tags);
        
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
    
    public void RecordQueueDepth(int deviceIndex, int depth)
    {
        lock (_queueDepths)
        {
            _queueDepths[deviceIndex] = depth;
        }
    }
    
    public void RecordGrainActivation(string grainType, TimeSpan duration)
    {
        var tags = new TagList
        {
            { "grain_type", grainType }
        };
        
        _grainActivations.Add(1, tags);
        _grainActivationLatency.Record(duration.TotalMilliseconds, tags);
    }
    
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
    
    // Observable gauge callbacks
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
    
    // Update methods for external systems to report metrics
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
    
    public void Dispose()
    {
        _meter?.Dispose();
        _activitySource?.Dispose();
    }
}

public enum TransferDirection
{
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    PeerToPeer
}