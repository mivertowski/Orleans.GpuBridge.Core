using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Orleans.GpuBridge.Abstractions.Exceptions;
using Orleans.GpuBridge.Resilience.Policies;

namespace Orleans.GpuBridge.Resilience.Chaos;

/// <summary>
/// Chaos engineering implementation for testing system resilience
/// </summary>
public sealed class ChaosEngineer : IChaosEngineer, IDisposable
{
    private readonly ILogger<ChaosEngineer> _logger;
    private readonly ChaosEngineeringOptions _options;
    private readonly Random _random;
    private readonly ThreadSafeRandom _threadSafeRandom;
    private long _totalOperations;
    private long _faultsInjected;
    private bool _disposed;

    public ChaosEngineer(
        ILogger<ChaosEngineer> logger,
        IOptions<GpuResiliencePolicyOptions> options)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options?.Value?.ChaosOptions ?? throw new ArgumentNullException(nameof(options));
        _random = new Random();
        _threadSafeRandom = new ThreadSafeRandom();
        
        _logger.LogInformation(
            "Chaos engineer initialized: Enabled={Enabled}, FaultProbability={FaultProbability:P}",
            _options.Enabled, _options.FaultInjectionProbability);
    }

    /// <summary>
    /// Executes an operation with potential chaos injection
    /// </summary>
    public async Task<T> ExecuteWithChaosAsync<T>(
        Func<CancellationToken, Task<T>> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        if (!_options.Enabled || _disposed)
        {
            return await operation(cancellationToken);
        }

        Interlocked.Increment(ref _totalOperations);

        // Pre-execution chaos injection
        await InjectPreExecutionChaosAsync(operationName, cancellationToken);

        try
        {
            var result = await operation(cancellationToken);
            
            // Post-execution chaos injection
            await InjectPostExecutionChaosAsync(operationName, cancellationToken);
            
            return result;
        }
        catch (Exception ex)
        {
            // Don't inject additional chaos if operation already failed
            _logger.LogDebug("Operation {OperationName} failed naturally: {Exception}", 
                operationName, ex.GetType().Name);
            throw;
        }
    }

    /// <summary>
    /// Executes an action with potential chaos injection
    /// </summary>
    public async Task ExecuteWithChaosAsync(
        Func<CancellationToken, Task> operation,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        await ExecuteWithChaosAsync(async ct =>
        {
            await operation(ct);
            return 0; // Dummy return value
        }, operationName, cancellationToken);
    }

    /// <summary>
    /// Injects latency into the operation
    /// </summary>
    public async Task InjectLatencyAsync(string operationName, CancellationToken cancellationToken = default)
    {
        if (!ShouldInjectFault(_options.LatencyInjection.InjectionProbability) || 
            !_options.LatencyInjection.Enabled)
        {
            return;
        }

        var latency = GenerateLatency();
        
        _logger.LogWarning(
            "[CHAOS] Injecting {Latency}ms latency into operation: {OperationName}",
            latency.TotalMilliseconds, operationName);

        await Task.Delay(latency, cancellationToken);
        
        Interlocked.Increment(ref _faultsInjected);
    }

    /// <summary>
    /// Injects exceptions into the operation
    /// </summary>
    public void InjectException(string operationName)
    {
        if (!ShouldInjectFault(_options.ExceptionInjection.InjectionProbability) || 
            !_options.ExceptionInjection.Enabled)
        {
            return;
        }

        var exceptionType = SelectRandomExceptionType();
        var exception = CreateException(exceptionType, operationName);
        
        _logger.LogWarning(
            "[CHAOS] Injecting {ExceptionType} into operation: {OperationName}",
            exceptionType, operationName);

        Interlocked.Increment(ref _faultsInjected);
        throw exception;
    }

    /// <summary>
    /// Simulates resource exhaustion
    /// </summary>
    public async Task<bool> SimulateResourceExhaustionAsync(
        string resourceType,
        string operationName,
        CancellationToken cancellationToken = default)
    {
        if (!_options.ResourceExhaustion.Enabled || _disposed)
        {
            return false;
        }

        var probability = resourceType.ToLowerInvariant() switch
        {
            "memory" => _options.ResourceExhaustion.MemoryExhaustionProbability,
            "compute" => _options.ResourceExhaustion.ComputeExhaustionProbability,
            _ => 0.0
        };

        if (!ShouldInjectFault(probability))
        {
            return false;
        }

        var duration = _options.ResourceExhaustion.ExhaustionDuration;
        
        _logger.LogWarning(
            "[CHAOS] Simulating {ResourceType} exhaustion for {Duration}ms in operation: {OperationName}",
            resourceType, duration.TotalMilliseconds, operationName);

        // Simulate resource exhaustion by consuming resources
        await SimulateResourceConsumptionAsync(resourceType, duration, cancellationToken);
        
        Interlocked.Increment(ref _faultsInjected);
        return true;
    }

    /// <summary>
    /// Gets chaos engineering metrics
    /// </summary>
    public ChaosMetrics GetMetrics()
    {
        return new ChaosMetrics(
            TotalOperations: _totalOperations,
            FaultsInjected: _faultsInjected,
            FaultInjectionRate: _totalOperations == 0 ? 0.0 : (double)_faultsInjected / _totalOperations,
            LatencyInjectionEnabled: _options.LatencyInjection.Enabled,
            ExceptionInjectionEnabled: _options.ExceptionInjection.Enabled,
            ResourceExhaustionEnabled: _options.ResourceExhaustion.Enabled);
    }

    /// <summary>
    /// Enables or disables chaos engineering
    /// </summary>
    public void SetEnabled(bool enabled)
    {
        _logger.LogInformation("Chaos engineering {Status}", enabled ? "enabled" : "disabled");
        // Note: This would require making _options mutable or using a different approach
        // For now, this is logged but the original options remain unchanged
    }

    /// <summary>
    /// Resets chaos engineering metrics
    /// </summary>
    public void ResetMetrics()
    {
        Interlocked.Exchange(ref _totalOperations, 0);
        Interlocked.Exchange(ref _faultsInjected, 0);
        
        _logger.LogInformation("Chaos engineering metrics reset");
    }

    /// <summary>
    /// Injects chaos before operation execution
    /// </summary>
    private async Task InjectPreExecutionChaosAsync(string operationName, CancellationToken cancellationToken)
    {
        // Latency injection
        await InjectLatencyAsync(operationName, cancellationToken);
        
        // Resource exhaustion simulation
        await SimulateResourceExhaustionAsync("memory", operationName, cancellationToken);
        await SimulateResourceExhaustionAsync("compute", operationName, cancellationToken);
        
        // Exception injection (throws if triggered)
        InjectException(operationName);
    }

    /// <summary>
    /// Injects chaos after operation execution
    /// </summary>
    private async Task InjectPostExecutionChaosAsync(string operationName, CancellationToken cancellationToken)
    {
        // Could add post-execution chaos here (e.g., corruption of results, delayed cleanup)
        await Task.CompletedTask;
    }

    /// <summary>
    /// Determines if a fault should be injected based on probability
    /// </summary>
    private bool ShouldInjectFault(double probability)
    {
        if (probability <= 0) return false;
        if (probability >= 1) return true;
        
        return _threadSafeRandom.NextDouble() < probability;
    }

    /// <summary>
    /// Generates random latency within configured bounds
    /// </summary>
    private TimeSpan GenerateLatency()
    {
        var minMs = _options.LatencyInjection.MinLatency.TotalMilliseconds;
        var maxMs = _options.LatencyInjection.MaxLatency.TotalMilliseconds;
        
        var latencyMs = minMs + _threadSafeRandom.NextDouble() * (maxMs - minMs);
        return TimeSpan.FromMilliseconds(latencyMs);
    }

    /// <summary>
    /// Selects a random exception type to inject
    /// </summary>
    private string SelectRandomExceptionType()
    {
        var types = _options.ExceptionInjection.ExceptionTypes;
        if (types.Length == 0)
        {
            return typeof(GpuOperationException).FullName!;
        }
        
        var index = _threadSafeRandom.Next(types.Length);
        return types[index];
    }

    /// <summary>
    /// Creates an exception instance of the specified type
    /// </summary>
    private Exception CreateException(string exceptionTypeName, string operationName)
    {
        return exceptionTypeName switch
        {
            var name when name.Contains("GpuOperationException") =>
                new GpuOperationException(operationName, $"[CHAOS] Injected fault in operation: {operationName}"),

            var name when name.Contains("GpuMemoryException") =>
                new GpuMemoryException($"[CHAOS] Injected memory fault in operation: {operationName}", 1024, 512),

            var name when name.Contains("GpuKernelException") =>
                new GpuKernelException($"[CHAOS] Injected kernel fault in operation: {operationName}", "chaos-kernel", operationName),

            var name when name.Contains("GpuDeviceException") =>
                new GpuDeviceException($"[CHAOS] Injected device fault in operation: {operationName}", -1, "chaos-device", "faulted", operationName),

            var name when name.Contains("TimeoutException") =>
                new TimeoutException($"[CHAOS] Injected timeout in operation: {operationName}"),

            _ => new GpuOperationException(operationName, $"[CHAOS] Injected unknown fault in operation: {operationName}")
        };
    }

    /// <summary>
    /// Simulates resource consumption for the specified duration
    /// </summary>
    private async Task SimulateResourceConsumptionAsync(string resourceType, TimeSpan duration, CancellationToken cancellationToken)
    {
        var endTime = DateTime.UtcNow + duration;
        
        // Different simulation strategies based on resource type
        switch (resourceType.ToLowerInvariant())
        {
            case "memory":
                await SimulateMemoryPressureAsync(endTime, cancellationToken);
                break;
            
            case "compute":
                await SimulateComputePressureAsync(endTime, cancellationToken);
                break;
        }
    }

    /// <summary>
    /// Simulates memory pressure
    /// </summary>
    private async Task SimulateMemoryPressureAsync(DateTime endTime, CancellationToken cancellationToken)
    {
        // Light simulation - just allocate some memory temporarily
        var allocations = new List<byte[]>();
        
        try
        {
            while (DateTime.UtcNow < endTime && !cancellationToken.IsCancellationRequested)
            {
                // Allocate small chunks to avoid actual OOM
                allocations.Add(new byte[1024]);
                await Task.Delay(10, cancellationToken);
            }
        }
        finally
        {
            allocations.Clear();
        }
    }

    /// <summary>
    /// Simulates compute pressure
    /// </summary>
    private async Task SimulateComputePressureAsync(DateTime endTime, CancellationToken cancellationToken)
    {
        var tasks = new List<Task>();
        
        // Start some CPU-intensive tasks
        for (int i = 0; i < Environment.ProcessorCount / 2; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                while (DateTime.UtcNow < endTime && !cancellationToken.IsCancellationRequested)
                {
                    // Light CPU work
                    Math.Sin(_threadSafeRandom.NextDouble() * Math.PI);
                    await Task.Delay(1, cancellationToken);
                }
            }, cancellationToken));
        }
        
        await Task.WhenAll(tasks);
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _disposed = true;
        _logger.LogInformation("Chaos engineer disposed");
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Interface for chaos engineering operations
/// </summary>
public interface IChaosEngineer
{
    /// <summary>
    /// Executes an operation with potential chaos injection
    /// </summary>
    Task<T> ExecuteWithChaosAsync<T>(Func<CancellationToken, Task<T>> operation, string operationName, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Executes an action with potential chaos injection
    /// </summary>
    Task ExecuteWithChaosAsync(Func<CancellationToken, Task> operation, string operationName, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets chaos engineering metrics
    /// </summary>
    ChaosMetrics GetMetrics();
    
    /// <summary>
    /// Enables or disables chaos engineering
    /// </summary>
    void SetEnabled(bool enabled);
    
    /// <summary>
    /// Resets metrics
    /// </summary>
    void ResetMetrics();
}

/// <summary>
/// Thread-safe random number generator
/// </summary>
internal sealed class ThreadSafeRandom
{
    private static readonly ThreadLocal<Random> ThreadLocalRandom = 
        new(() => new Random(Guid.NewGuid().GetHashCode()));

    public double NextDouble() => ThreadLocalRandom.Value!.NextDouble();
    public int Next(int maxValue) => ThreadLocalRandom.Value!.Next(maxValue);
}

/// <summary>
/// Chaos engineering metrics
/// </summary>
public readonly record struct ChaosMetrics(
    long TotalOperations,
    long FaultsInjected,
    double FaultInjectionRate,
    bool LatencyInjectionEnabled,
    bool ExceptionInjectionEnabled,
    bool ResourceExhaustionEnabled);