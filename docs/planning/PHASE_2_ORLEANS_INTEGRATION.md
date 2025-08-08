# Phase 2: Orleans Integration Implementation Specification

## Duration: Weeks 3-4

## Objectives
Integrate Orleans.GpuBridge with the Orleans framework, implementing GPU-aware placement strategies, grain infrastructure, and streaming capabilities.

## Week 3: Placement Strategy & Core Grains

### Day 1-2: GPU Placement Strategy

#### GpuPlacementStrategy Implementation
```csharp
namespace Orleans.GpuBridge.Runtime.Placement;

[Serializable]
public sealed class GpuPlacementStrategy : PlacementStrategy
{
    public static GpuPlacementStrategy Instance { get; } = new();
    
    public bool PreferLocalPlacement { get; init; }
    public int MinimumGpuMemoryMB { get; init; }
    public GpuBackend PreferredBackend { get; init; } = GpuBackend.Cuda;
}

[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class GpuPlacementAttribute : PlacementAttribute
{
    public GpuPlacementAttribute() : base(GpuPlacementStrategy.Instance) { }
    
    public bool PreferLocal { get; set; }
    public int MinMemoryMB { get; set; }
}
```

#### GpuPlacementDirector Implementation
```csharp
public sealed class GpuPlacementDirector : IPlacementDirector
{
    private readonly ILocalSiloDetails _localSilo;
    private readonly IGpuCapacityGrain _capacityGrain;
    
    public GpuPlacementDirector(
        ILocalSiloDetails localSilo,
        IGrainFactory grainFactory)
    {
        _localSilo = localSilo;
        _capacityGrain = grainFactory.GetGrain<IGpuCapacityGrain>(0);
    }
    
    public async Task<SiloAddress> OnAddActivation(
        PlacementStrategy strategy,
        PlacementTarget target,
        IPlacementContext context)
    {
        if (strategy is not GpuPlacementStrategy gpuStrategy)
        {
            throw new InvalidOperationException("Invalid strategy type");
        }
        
        // Get GPU-capable silos
        var gpuSilos = await _capacityGrain.GetGpuCapableSilosAsync();
        
        if (gpuSilos.Count == 0)
        {
            // Fallback to any available silo
            return context.GetCompatibleSilos(target).FirstOrDefault()
                ?? throw new OrleansException("No compatible silos found");
        }
        
        // Prefer local if requested and capable
        if (gpuStrategy.PreferLocalPlacement)
        {
            var localCapacity = gpuSilos.FirstOrDefault(
                s => s.SiloAddress.Equals(_localSilo.SiloAddress));
            
            if (localCapacity != null && 
                localCapacity.AvailableMemoryMB >= gpuStrategy.MinimumGpuMemoryMB)
            {
                return _localSilo.SiloAddress;
            }
        }
        
        // Select silo with most available GPU memory
        var bestSilo = gpuSilos
            .Where(s => s.AvailableMemoryMB >= gpuStrategy.MinimumGpuMemoryMB)
            .OrderByDescending(s => s.AvailableMemoryMB)
            .ThenBy(s => s.QueueDepth)
            .FirstOrDefault();
        
        if (bestSilo != null)
        {
            return bestSilo.SiloAddress;
        }
        
        // Final fallback
        return context.GetCompatibleSilos(target).FirstOrDefault()
            ?? throw new OrleansException("No compatible silos found");
    }
}
```

#### GPU Capacity Tracking
```csharp
public interface IGpuCapacityGrain : IGrainWithIntegerKey
{
    Task RegisterSiloAsync(SiloAddress silo, GpuCapacity capacity);
    Task UnregisterSiloAsync(SiloAddress silo);
    Task UpdateCapacityAsync(SiloAddress silo, GpuCapacity capacity);
    Task<List<SiloGpuCapacity>> GetGpuCapableSilosAsync();
}

[Reentrant]
public sealed class GpuCapacityGrain : Grain, IGpuCapacityGrain
{
    private readonly Dictionary<SiloAddress, GpuCapacity> _capacities = new();
    private readonly ILogger<GpuCapacityGrain> _logger;
    
    public GpuCapacityGrain(ILogger<GpuCapacityGrain> logger) => _logger = logger;
    
    public Task RegisterSiloAsync(SiloAddress silo, GpuCapacity capacity)
    {
        _capacities[silo] = capacity;
        _logger.LogInformation(
            "Registered GPU silo {Silo} with {Devices} devices, {Memory}MB memory",
            silo, capacity.DeviceCount, capacity.TotalMemoryMB);
        return Task.CompletedTask;
    }
    
    public Task UnregisterSiloAsync(SiloAddress silo)
    {
        _capacities.Remove(silo);
        _logger.LogInformation("Unregistered GPU silo {Silo}", silo);
        return Task.CompletedTask;
    }
    
    public Task UpdateCapacityAsync(SiloAddress silo, GpuCapacity capacity)
    {
        if (_capacities.ContainsKey(silo))
        {
            _capacities[silo] = capacity;
        }
        return Task.CompletedTask;
    }
    
    public Task<List<SiloGpuCapacity>> GetGpuCapableSilosAsync()
    {
        var result = _capacities
            .Select(kvp => new SiloGpuCapacity(kvp.Key, kvp.Value))
            .ToList();
        return Task.FromResult(result);
    }
}

public sealed record GpuCapacity(
    int DeviceCount,
    long TotalMemoryMB,
    long AvailableMemoryMB,
    int QueueDepth,
    GpuBackend Backend);

public sealed record SiloGpuCapacity(
    SiloAddress SiloAddress,
    GpuCapacity Capacity)
{
    public long AvailableMemoryMB => Capacity.AvailableMemoryMB;
    public int QueueDepth => Capacity.QueueDepth;
}
```

### Day 3-4: GpuBatchGrain Implementation

#### Enhanced GpuBatchGrain
```csharp
namespace Orleans.GpuBridge.Grains;

public interface IGpuBatchGrain<TIn, TOut> : IGrainWithStringKey
    where TIn : notnull where TOut : notnull
{
    Task<GpuBatchResult<TOut>> ExecuteAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints = null);
    
    Task<GpuBatchResult<TOut>> ExecuteWithCallbackAsync(
        IReadOnlyList<TIn> batch,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints = null);
}

[GpuPlacement(PreferLocal = true)]
[StatelessWorker(1)] // One per silo
public sealed class GpuBatchGrain<TIn, TOut> : Grain, IGpuBatchGrain<TIn, TOut>
    where TIn : notnull where TOut : notnull
{
    private readonly ILogger<GpuBatchGrain<TIn, TOut>> _logger;
    private IGpuBridge _bridge = default!;
    private IGpuKernel<TIn, TOut> _kernel = default!;
    private KernelId _kernelId = default!;
    private readonly SemaphoreSlim _concurrencyLimit;
    
    public GpuBatchGrain(ILogger<GpuBatchGrain<TIn, TOut>> logger)
    {
        _logger = logger;
        _concurrencyLimit = new SemaphoreSlim(
            Environment.ProcessorCount * 2);
    }
    
    public override async Task OnActivateAsync(CancellationToken ct)
    {
        _kernelId = KernelId.Parse(this.GetPrimaryKeyString());
        _bridge = ServiceProvider.GetRequiredService<IGpuBridge>();
        _kernel = await _bridge.GetKernelAsync<TIn, TOut>(_kernelId, ct);
        
        _logger.LogInformation(
            "Activated GPU batch grain for kernel {KernelId}",
            _kernelId);
    }
    
    public async Task<GpuBatchResult<TOut>> ExecuteAsync(
        IReadOnlyList<TIn> batch,
        GpuExecutionHints? hints = null)
    {
        await _concurrencyLimit.WaitAsync();
        try
        {
            var stopwatch = Stopwatch.StartNew();
            
            // Submit batch to kernel
            var handle = await _kernel.SubmitBatchAsync(batch, hints);
            
            // Collect results
            var results = new List<TOut>();
            await foreach (var result in _kernel.ReadResultsAsync(handle))
            {
                results.Add(result);
            }
            
            stopwatch.Stop();
            
            return new GpuBatchResult<TOut>(
                results,
                stopwatch.Elapsed,
                handle.Id,
                _kernelId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, 
                "Failed to execute batch on kernel {KernelId}",
                _kernelId);
            
            return new GpuBatchResult<TOut>(
                Array.Empty<TOut>(),
                TimeSpan.Zero,
                string.Empty,
                _kernelId,
                ex.Message);
        }
        finally
        {
            _concurrencyLimit.Release();
        }
    }
    
    public async Task<GpuBatchResult<TOut>> ExecuteWithCallbackAsync(
        IReadOnlyList<TIn> batch,
        IGpuResultObserver<TOut> observer,
        GpuExecutionHints? hints = null)
    {
        var result = await ExecuteAsync(batch, hints);
        
        // Stream results to observer
        foreach (var item in result.Results)
        {
            await observer.OnNextAsync(item);
        }
        
        await observer.OnCompletedAsync();
        
        return result;
    }
}

public sealed record GpuBatchResult<TOut>(
    IReadOnlyList<TOut> Results,
    TimeSpan ExecutionTime,
    string HandleId,
    KernelId KernelId,
    string? Error = null)
{
    public bool Success => Error == null;
}

public interface IGpuResultObserver<T>
{
    Task OnNextAsync(T item);
    Task OnErrorAsync(Exception error);
    Task OnCompletedAsync();
}
```

### Day 5: GpuStreamGrain Implementation

#### Stream Processing Grain
```csharp
public interface IGpuStreamGrain<TIn, TOut> : IGrainWithStringKey
    where TIn : notnull where TOut : notnull
{
    Task StartProcessingAsync(
        StreamId inputStream,
        StreamId outputStream,
        GpuExecutionHints? hints = null);
    
    Task StopProcessingAsync();
    
    Task<StreamProcessingStatus> GetStatusAsync();
}

[GpuPlacement]
public sealed class GpuStreamGrain<TIn, TOut> : Grain, IGpuStreamGrain<TIn, TOut>
    where TIn : notnull where TOut : notnull
{
    private readonly ILogger<GpuStreamGrain<TIn, TOut>> _logger;
    private IGpuKernel<TIn, TOut> _kernel = default!;
    private IAsyncStream<TIn> _inputStream = default!;
    private IAsyncStream<TOut> _outputStream = default!;
    private StreamSubscriptionHandle<TIn> _subscription = default!;
    private readonly Channel<TIn> _buffer;
    private CancellationTokenSource _cts = default!;
    private Task _processingTask = default!;
    private StreamProcessingStatus _status = StreamProcessingStatus.Idle;
    
    public GpuStreamGrain(ILogger<GpuStreamGrain<TIn, TOut>> logger)
    {
        _logger = logger;
        _buffer = Channel.CreateUnbounded<TIn>(
            new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = false
            });
    }
    
    public async Task StartProcessingAsync(
        StreamId inputStream,
        StreamId outputStream,
        GpuExecutionHints? hints = null)
    {
        if (_status == StreamProcessingStatus.Processing)
        {
            throw new InvalidOperationException("Already processing");
        }
        
        var kernelId = KernelId.Parse(this.GetPrimaryKeyString());
        var bridge = ServiceProvider.GetRequiredService<IGpuBridge>();
        _kernel = await bridge.GetKernelAsync<TIn, TOut>(kernelId);
        
        var streamProvider = this.GetStreamProvider("Default");
        _inputStream = streamProvider.GetStream<TIn>(inputStream);
        _outputStream = streamProvider.GetStream<TOut>(outputStream);
        
        // Subscribe to input stream
        _subscription = await _inputStream.SubscribeAsync(
            async (item, token) =>
            {
                await _buffer.Writer.WriteAsync(item, token);
            });
        
        // Start processing loop
        _cts = new CancellationTokenSource();
        _processingTask = ProcessStreamAsync(hints, _cts.Token);
        _status = StreamProcessingStatus.Processing;
        
        _logger.LogInformation(
            "Started stream processing from {Input} to {Output}",
            inputStream, outputStream);
    }
    
    public async Task StopProcessingAsync()
    {
        if (_status != StreamProcessingStatus.Processing)
        {
            return;
        }
        
        _status = StreamProcessingStatus.Stopping;
        
        // Unsubscribe from input
        if (_subscription != null)
        {
            await _subscription.UnsubscribeAsync();
        }
        
        // Signal completion
        _buffer.Writer.TryComplete();
        
        // Cancel processing
        _cts?.Cancel();
        
        // Wait for processing to complete
        if (_processingTask != null)
        {
            await _processingTask;
        }
        
        _status = StreamProcessingStatus.Idle;
        
        _logger.LogInformation("Stopped stream processing");
    }
    
    public Task<StreamProcessingStatus> GetStatusAsync()
    {
        return Task.FromResult(_status);
    }
    
    private async Task ProcessStreamAsync(
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        const int batchSize = hints?.MaxMicroBatch ?? 128;
        var batch = new List<TIn>(batchSize);
        var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(100));
        
        try
        {
            while (!ct.IsCancellationRequested)
            {
                // Collect batch
                while (batch.Count < batchSize && 
                       _buffer.Reader.TryRead(out var item))
                {
                    batch.Add(item);
                }
                
                // Process if we have items or timeout
                if (batch.Count > 0 && 
                    (batch.Count >= batchSize || await timer.WaitForNextTickAsync(ct)))
                {
                    await ProcessBatchAsync(batch, hints, ct);
                    batch.Clear();
                }
                
                // Small delay to prevent tight loop
                if (batch.Count == 0)
                {
                    await Task.Delay(10, ct);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Expected on shutdown
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Stream processing failed");
            _status = StreamProcessingStatus.Failed;
        }
        finally
        {
            timer.Dispose();
        }
    }
    
    private async Task ProcessBatchAsync(
        List<TIn> batch,
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        try
        {
            var handle = await _kernel.SubmitBatchAsync(batch, hints, ct);
            
            await foreach (var result in _kernel.ReadResultsAsync(handle, ct))
            {
                await _outputStream.OnNextAsync(result);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, 
                "Failed to process batch of {Count} items",
                batch.Count);
        }
    }
}

public enum StreamProcessingStatus
{
    Idle,
    Processing,
    Stopping,
    Failed
}
```

## Week 4: Resident Grains & Orleans Streams

### Day 6-7: GpuResidentGrain Implementation

#### Stateful GPU Grain
```csharp
public interface IGpuResidentGrain<TState> : IGrainWithStringKey
    where TState : class, new()
{
    Task<TState> GetStateAsync();
    Task SetStateAsync(TState state);
    Task<TResult> ComputeAsync<TResult>(
        Func<TState, TResult> computation,
        GpuExecutionHints? hints = null);
    Task<bool> EvictFromGpuAsync();
    Task<bool> LoadToGpuAsync();
}

[GpuPlacement(MinMemoryMB = 1024)]
public sealed class GpuResidentGrain<TState> : Grain, IGpuResidentGrain<TState>
    where TState : class, new()
{
    private readonly ILogger<GpuResidentGrain<TState>> _logger;
    private readonly IPersistentState<TState> _state;
    private IGpuMemory<byte> _gpuMemory = default!;
    private TState _cachedState = default!;
    private bool _isResident;
    private DateTime _lastAccess;
    
    public GpuResidentGrain(
        [PersistentState("state", "gpu-storage")]
        IPersistentState<TState> state,
        ILogger<GpuResidentGrain<TState>> logger)
    {
        _state = state;
        _logger = logger;
    }
    
    public override async Task OnActivateAsync(CancellationToken ct)
    {
        await _state.ReadStateAsync();
        _cachedState = _state.State ?? new TState();
        _lastAccess = DateTime.UtcNow;
        
        // Try to load into GPU memory
        await LoadToGpuAsync();
        
        // Register timer for eviction check
        RegisterTimer(
            CheckEvictionAsync,
            null,
            TimeSpan.FromMinutes(5),
            TimeSpan.FromMinutes(5));
    }
    
    public Task<TState> GetStateAsync()
    {
        _lastAccess = DateTime.UtcNow;
        return Task.FromResult(_cachedState);
    }
    
    public async Task SetStateAsync(TState state)
    {
        _cachedState = state;
        _state.State = state;
        await _state.WriteStateAsync();
        _lastAccess = DateTime.UtcNow;
        
        // Update GPU memory if resident
        if (_isResident)
        {
            await UpdateGpuMemoryAsync();
        }
    }
    
    public async Task<TResult> ComputeAsync<TResult>(
        Func<TState, TResult> computation,
        GpuExecutionHints? hints = null)
    {
        _lastAccess = DateTime.UtcNow;
        
        if (!_isResident)
        {
            await LoadToGpuAsync();
        }
        
        // This would normally dispatch to GPU
        // For now, CPU fallback
        return await Task.Run(() => computation(_cachedState));
    }
    
    public Task<bool> LoadToGpuAsync()
    {
        if (_isResident) return Task.FromResult(true);
        
        try
        {
            var pool = ServiceProvider
                .GetRequiredService<IGpuMemoryPool<byte>>();
            
            var bytes = SerializeState(_cachedState);
            _gpuMemory = pool.Rent(bytes.Length);
            
            bytes.CopyTo(_gpuMemory.AsMemory());
            _gpuMemory.CopyToDeviceAsync().Wait();
            
            _isResident = true;
            _logger.LogInformation(
                "Loaded {Size} bytes to GPU memory",
                bytes.Length);
            
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load state to GPU");
            return Task.FromResult(false);
        }
    }
    
    public Task<bool> EvictFromGpuAsync()
    {
        if (!_isResident) return Task.FromResult(false);
        
        try
        {
            _gpuMemory?.Dispose();
            _gpuMemory = null!;
            _isResident = false;
            
            _logger.LogInformation("Evicted state from GPU memory");
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to evict from GPU");
            return Task.FromResult(false);
        }
    }
    
    private async Task CheckEvictionAsync(object? state)
    {
        var idleTime = DateTime.UtcNow - _lastAccess;
        if (idleTime > TimeSpan.FromMinutes(10))
        {
            await EvictFromGpuAsync();
        }
    }
    
    private async Task UpdateGpuMemoryAsync()
    {
        if (_gpuMemory != null)
        {
            var bytes = SerializeState(_cachedState);
            bytes.CopyTo(_gpuMemory.AsMemory());
            await _gpuMemory.CopyToDeviceAsync();
        }
    }
    
    private static byte[] SerializeState(TState state)
    {
        // Use MessagePack or similar for efficiency
        return JsonSerializer.SerializeToUtf8Bytes(state);
    }
}
```

### Day 8-9: Orleans Streams Integration

#### GPU Result Stream Provider
```csharp
namespace Orleans.GpuBridge.Streams;

public class GpuResultStreamProvider : IStreamProvider
{
    private readonly string _name;
    private readonly IGrainFactory _grainFactory;
    private readonly ILogger<GpuResultStreamProvider> _logger;
    
    public string Name => _name;
    public bool IsRewindable => false;
    
    public GpuResultStreamProvider(
        string name,
        IGrainFactory grainFactory,
        ILogger<GpuResultStreamProvider> logger)
    {
        _name = name;
        _grainFactory = grainFactory;
        _logger = logger;
    }
    
    public IAsyncStream<T> GetStream<T>(StreamId streamId)
    {
        return new GpuResultStream<T>(streamId, this);
    }
    
    // Implementation of other IStreamProvider methods...
}

internal class GpuResultStream<T> : IAsyncStream<T>
{
    private readonly StreamId _streamId;
    private readonly GpuResultStreamProvider _provider;
    private readonly Channel<T> _channel;
    
    public GpuResultStream(StreamId streamId, GpuResultStreamProvider provider)
    {
        _streamId = streamId;
        _provider = provider;
        _channel = Channel.CreateUnbounded<T>();
    }
    
    public async Task OnNextAsync(T item, StreamSequenceToken? token = null)
    {
        await _channel.Writer.WriteAsync(item);
    }
    
    public async Task<StreamSubscriptionHandle<T>> SubscribeAsync(
        IAsyncObserver<T> observer,
        StreamSequenceToken? token = null,
        StreamFilterPredicate? filterFunc = null,
        object? filterData = null)
    {
        var subscription = new GpuStreamSubscription<T>(
            _streamId,
            observer,
            _channel.Reader);
        
        _ = subscription.StartAsync();
        
        return subscription;
    }
    
    // Other interface implementations...
}
```

### Day 10: Integration Testing

#### Orleans TestingHost Tests
```csharp
public class GpuGrainIntegrationTests : IAsyncLifetime
{
    private TestCluster _cluster = default!;
    
    public async Task InitializeAsync()
    {
        var builder = new TestClusterBuilder();
        builder.AddSiloBuilderConfigurator<TestSiloConfigurator>();
        _cluster = builder.Build();
        await _cluster.DeployAsync();
    }
    
    public async Task DisposeAsync()
    {
        await _cluster.StopAllSilosAsync();
    }
    
    [Fact]
    public async Task BatchGrain_Should_Process_Work()
    {
        var grain = _cluster.GrainFactory
            .GetGrain<IGpuBatchGrain<float[], float>>("kernels/sum");
        
        var input = new[]
        {
            new[] { 1f, 2f, 3f },
            new[] { 4f, 5f, 6f }
        };
        
        var result = await grain.ExecuteAsync(input);
        
        Assert.True(result.Success);
        Assert.Equal(2, result.Results.Count);
        Assert.Equal(6f, result.Results[0]); // 1+2+3
        Assert.Equal(15f, result.Results[1]); // 4+5+6
    }
    
    [Fact]
    public async Task StreamGrain_Should_Process_Stream()
    {
        var streamProvider = _cluster.Client.GetStreamProvider("Default");
        var inputStream = streamProvider.GetStream<float[]>(
            StreamId.Create("input", Guid.NewGuid()));
        var outputStream = streamProvider.GetStream<float>(
            StreamId.Create("output", Guid.NewGuid()));
        
        var results = new List<float>();
        await outputStream.SubscribeAsync((value, token) =>
        {
            results.Add(value);
            return Task.CompletedTask;
        });
        
        var grain = _cluster.GrainFactory
            .GetGrain<IGpuStreamGrain<float[], float>>("kernels/sum");
        
        await grain.StartProcessingAsync(
            inputStream.StreamId,
            outputStream.StreamId);
        
        // Send data
        await inputStream.OnNextAsync(new[] { 1f, 2f, 3f });
        await inputStream.OnNextAsync(new[] { 4f, 5f, 6f });
        
        // Wait for processing
        await Task.Delay(500);
        
        Assert.Contains(6f, results);
        Assert.Contains(15f, results);
        
        await grain.StopProcessingAsync();
    }
    
    private class TestSiloConfigurator : ISiloConfigurator
    {
        public void Configure(ISiloBuilder siloBuilder)
        {
            siloBuilder.Services.AddGpuBridge(options =>
            {
                options.PreferGpu = false; // Use CPU for tests
            })
            .AddKernel(k => k
                .Id("kernels/sum")
                .Input<float[]>()
                .Output<float>()
                .WithFactory(sp => new SumKernel()));
            
            siloBuilder.AddMemoryGrainStorageAsDefault();
            siloBuilder.AddMemoryStreams("Default");
        }
    }
}
```

## Deliverables Checklist

### Week 3 Deliverables
- [ ] GPU placement strategy implementation
- [ ] Placement director with silo selection logic
- [ ] GPU capacity tracking grain
- [ ] Enhanced GpuBatchGrain with concurrency control
- [ ] GpuStreamGrain for stream processing
- [ ] Unit tests for placement logic

### Week 4 Deliverables
- [ ] GpuResidentGrain with state management
- [ ] GPU memory management for resident state
- [ ] Orleans Streams integration
- [ ] Custom stream provider for GPU results
- [ ] Integration tests with TestingHost
- [ ] Performance benchmarks

## Success Metrics

### Functional Requirements
- Grains activate on GPU-capable silos
- Automatic fallback to CPU silos when GPU unavailable
- Stream processing maintains order
- State persistence works correctly

### Performance Targets
- Grain activation < 100ms
- Stream latency < 10ms per item
- Batch processing scales linearly with size
- Memory usage stable under load

### Integration Quality
- All Orleans patterns properly implemented
- Clean separation of concerns
- Proper lifecycle management
- No memory leaks

## Testing Strategy

### Unit Testing
- Placement strategy logic
- Grain activation lifecycle
- Stream subscription/unsubscription
- State management

### Integration Testing
- Multi-silo cluster scenarios
- GPU/CPU fallback behavior
- Stream end-to-end processing
- Grain migration on failure

### Performance Testing
- Throughput benchmarks
- Latency measurements
- Memory pressure tests
- Concurrent grain activation

## Next Phase Preview

Phase 3 will implement actual GPU execution:
- DotCompute backend integration
- Persistent kernel implementation
- Ring buffer communication
- GPU memory management
- Hardware-specific optimizations