# Phase 3: GPU Runtime Implementation Specification

## Duration: Weeks 5-6

## Objectives
Implement actual GPU execution through DotCompute integration, persistent kernel hosts with ring buffer communication, and comprehensive GPU memory management.

## Week 5: DotCompute Integration & Device Management

### Day 1-2: DotCompute Adapter Foundation

#### DotComputeAdapter Implementation
```csharp
namespace Orleans.GpuBridge.DotCompute;

public sealed class DotComputeAdapter : IGpuBackendAdapter
{
    private readonly ILogger<DotComputeAdapter> _logger;
    private readonly DotComputeRuntime _runtime;
    private readonly Dictionary<int, DeviceContext> _devices;
    
    public DotComputeAdapter(
        ILogger<DotComputeAdapter> logger,
        IOptions<DotComputeOptions> options)
    {
        _logger = logger;
        _runtime = new DotComputeRuntime(options.Value);
        _devices = new Dictionary<int, DeviceContext>();
    }
    
    public async Task InitializeAsync(CancellationToken ct)
    {
        _logger.LogInformation("Initializing DotCompute adapter");
        
        // Initialize runtime
        await _runtime.InitializeAsync();
        
        // Enumerate devices
        var devices = await _runtime.EnumerateDevicesAsync();
        
        foreach (var device in devices)
        {
            var context = await CreateDeviceContextAsync(device);
            _devices[device.Index] = context;
            
            _logger.LogInformation(
                "Initialized device {Index}: {Name} ({Memory}MB, {ComputeUnits} CUs)",
                device.Index, device.Name, 
                device.TotalMemoryMB, device.ComputeUnits);
        }
    }
    
    public async Task<ICompiledKernel> CompileKernelAsync(
        string source,
        KernelCompilationOptions options,
        CancellationToken ct)
    {
        _logger.LogDebug("Compiling kernel: {Name}", options.Name);
        
        var compiledKernel = await _runtime.CompileAsync(
            source,
            new DotCompute.CompilationOptions
            {
                TargetArchitecture = options.TargetArch,
                OptimizationLevel = options.OptLevel,
                EnableDebugInfo = options.Debug,
                AotMode = true // Always AOT for production
            },
            ct);
        
        return new DotComputeKernel(compiledKernel, _runtime);
    }
    
    public async Task<KernelExecutionHandle> LaunchKernelAsync(
        ICompiledKernel kernel,
        KernelLaunchParameters parameters,
        CancellationToken ct)
    {
        var device = SelectDevice(parameters.PreferredDevice);
        var context = _devices[device];
        
        // Allocate buffers
        var inputBuffer = await context.AllocateAsync(
            parameters.InputSize, 
            BufferFlags.ReadOnly);
        var outputBuffer = await context.AllocateAsync(
            parameters.OutputSize,
            BufferFlags.WriteOnly);
        
        // Copy input data
        await inputBuffer.WriteAsync(parameters.InputData, ct);
        
        // Configure launch
        var config = new LaunchConfiguration
        {
            GlobalWorkSize = parameters.GlobalWorkSize,
            LocalWorkSize = parameters.LocalWorkSize,
            SharedMemoryBytes = parameters.SharedMemoryBytes
        };
        
        // Launch kernel
        var dcKernel = (DotComputeKernel)kernel;
        var event = await context.LaunchAsync(
            dcKernel.Compiled,
            config,
            inputBuffer,
            outputBuffer,
            ct);
        
        return new KernelExecutionHandle(
            Guid.NewGuid().ToString(),
            event,
            outputBuffer,
            context);
    }
    
    private async Task<DeviceContext> CreateDeviceContextAsync(
        DotCompute.Device device)
    {
        var context = await _runtime.CreateContextAsync(device);
        
        // Create command queues
        var computeQueue = await context.CreateQueueAsync(
            QueueFlags.OutOfOrder | QueueFlags.ProfilingEnabled);
        var transferQueue = await context.CreateQueueAsync(
            QueueFlags.InOrder);
        
        return new DeviceContext(
            device,
            context,
            computeQueue,
            transferQueue);
    }
    
    private int SelectDevice(int? preferred)
    {
        if (preferred.HasValue && _devices.ContainsKey(preferred.Value))
        {
            return preferred.Value;
        }
        
        // Select device with most available memory
        return _devices
            .OrderByDescending(kvp => kvp.Value.AvailableMemory)
            .First()
            .Key;
    }
}

internal sealed class DeviceContext
{
    public DotCompute.Device Device { get; }
    public DotCompute.Context Context { get; }
    public DotCompute.CommandQueue ComputeQueue { get; }
    public DotCompute.CommandQueue TransferQueue { get; }
    public long AvailableMemory { get; private set; }
    
    public DeviceContext(
        DotCompute.Device device,
        DotCompute.Context context,
        DotCompute.CommandQueue computeQueue,
        DotCompute.CommandQueue transferQueue)
    {
        Device = device;
        Context = context;
        ComputeQueue = computeQueue;
        TransferQueue = transferQueue;
        AvailableMemory = device.TotalMemoryBytes;
    }
    
    public async Task<DotCompute.Buffer> AllocateAsync(
        long sizeBytes,
        BufferFlags flags)
    {
        var buffer = await Context.CreateBufferAsync(sizeBytes, flags);
        AvailableMemory -= sizeBytes;
        return buffer;
    }
    
    public void Release(DotCompute.Buffer buffer)
    {
        AvailableMemory += buffer.SizeBytes;
        buffer.Dispose();
    }
}
```

### Day 3-4: Enhanced Device Broker

#### DeviceBroker with Resource Management
```csharp
namespace Orleans.GpuBridge.Runtime;

public sealed class DeviceBroker : IDeviceBroker
{
    private readonly ILogger<DeviceBroker> _logger;
    private readonly IGpuBackendAdapter _adapter;
    private readonly List<GpuDevice> _devices;
    private readonly Dictionary<int, DeviceWorkQueue> _queues;
    private readonly SemaphoreSlim _allocationLock;
    
    public DeviceBroker(
        ILogger<DeviceBroker> logger,
        IGpuBackendAdapter adapter)
    {
        _logger = logger;
        _adapter = adapter;
        _devices = new List<GpuDevice>();
        _queues = new Dictionary<int, DeviceWorkQueue>();
        _allocationLock = new SemaphoreSlim(1, 1);
    }
    
    public async Task InitializeAsync(CancellationToken ct)
    {
        _logger.LogInformation("Initializing device broker");
        
        await _adapter.InitializeAsync(ct);
        
        var devices = await _adapter.GetDevicesAsync(ct);
        
        foreach (var device in devices)
        {
            _devices.Add(device);
            _queues[device.Index] = new DeviceWorkQueue(device, _logger);
            
            _logger.LogInformation(
                "Registered device {Index}: {Name}",
                device.Index, device.Name);
        }
        
        // Start queue processors
        foreach (var queue in _queues.Values)
        {
            _ = queue.StartProcessingAsync(ct);
        }
    }
    
    public async Task<DeviceAllocation> AllocateDeviceAsync(
        AllocationRequest request,
        CancellationToken ct)
    {
        await _allocationLock.WaitAsync(ct);
        try
        {
            // Find best device
            var device = SelectBestDevice(request);
            if (device == null)
            {
                throw new InsufficientResourcesException(
                    "No suitable GPU device available");
            }
            
            // Reserve resources
            var allocation = new DeviceAllocation(
                Guid.NewGuid(),
                device,
                request.MemoryBytes,
                DateTime.UtcNow);
            
            device.ReserveMemory(request.MemoryBytes);
            
            _logger.LogDebug(
                "Allocated {Memory}MB on device {Device}",
                request.MemoryBytes / 1024 / 1024,
                device.Index);
            
            return allocation;
        }
        finally
        {
            _allocationLock.Release();
        }
    }
    
    public Task ReleaseAllocationAsync(
        DeviceAllocation allocation,
        CancellationToken ct)
    {
        if (_devices.FirstOrDefault(d => d.Index == allocation.Device.Index) 
            is GpuDevice device)
        {
            device.ReleaseMemory(allocation.MemoryBytes);
            
            _logger.LogDebug(
                "Released {Memory}MB on device {Device}",
                allocation.MemoryBytes / 1024 / 1024,
                device.Index);
        }
        
        return Task.CompletedTask;
    }
    
    public async Task<WorkHandle> EnqueueWorkAsync(
        int deviceIndex,
        IKernelWork work,
        CancellationToken ct)
    {
        if (!_queues.TryGetValue(deviceIndex, out var queue))
        {
            throw new ArgumentException(
                $"Device {deviceIndex} not found");
        }
        
        return await queue.EnqueueAsync(work, ct);
    }
    
    public GpuDeviceStats GetDeviceStats(int deviceIndex)
    {
        if (!_devices.Any(d => d.Index == deviceIndex))
        {
            throw new ArgumentException(
                $"Device {deviceIndex} not found");
        }
        
        var device = _devices[deviceIndex];
        var queue = _queues[deviceIndex];
        
        return new GpuDeviceStats(
            device.Index,
            device.Name,
            device.TotalMemoryBytes,
            device.AvailableMemoryBytes,
            queue.QueueDepth,
            queue.ProcessedCount,
            queue.FailedCount,
            queue.AverageLatencyMs);
    }
    
    private GpuDevice? SelectBestDevice(AllocationRequest request)
    {
        return _devices
            .Where(d => d.AvailableMemoryBytes >= request.MemoryBytes)
            .Where(d => !request.RequiredCapabilities.Any() || 
                        request.RequiredCapabilities.All(
                            cap => d.Capabilities.Contains(cap)))
            .OrderByDescending(d => d.AvailableMemoryBytes)
            .ThenBy(d => _queues[d.Index].QueueDepth)
            .FirstOrDefault();
    }
}

internal sealed class DeviceWorkQueue
{
    private readonly GpuDevice _device;
    private readonly ILogger _logger;
    private readonly Channel<WorkItem> _queue;
    private long _processedCount;
    private long _failedCount;
    private readonly MovingAverage _latency;
    
    public int QueueDepth => _queue.Reader.Count;
    public long ProcessedCount => _processedCount;
    public long FailedCount => _failedCount;
    public double AverageLatencyMs => _latency.Average;
    
    public DeviceWorkQueue(GpuDevice device, ILogger logger)
    {
        _device = device;
        _logger = logger;
        _queue = Channel.CreateUnbounded<WorkItem>();
        _latency = new MovingAverage(100);
    }
    
    public async Task<WorkHandle> EnqueueAsync(
        IKernelWork work,
        CancellationToken ct)
    {
        var handle = new WorkHandle(Guid.NewGuid());
        var item = new WorkItem(handle, work, DateTime.UtcNow);
        
        await _queue.Writer.WriteAsync(item, ct);
        
        return handle;
    }
    
    public async Task StartProcessingAsync(CancellationToken ct)
    {
        await foreach (var item in _queue.Reader.ReadAllAsync(ct))
        {
            try
            {
                var sw = Stopwatch.StartNew();
                await item.Work.ExecuteAsync(_device, ct);
                sw.Stop();
                
                _latency.Add(sw.ElapsedMilliseconds);
                Interlocked.Increment(ref _processedCount);
                
                item.Handle.Complete(sw.Elapsed);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "Failed to execute work on device {Device}",
                    _device.Index);
                
                Interlocked.Increment(ref _failedCount);
                item.Handle.Fail(ex);
            }
        }
    }
    
    private record WorkItem(
        WorkHandle Handle,
        IKernelWork Work,
        DateTime EnqueuedAt);
}
```

### Day 5: Memory Pool Management

#### GPU Memory Pool Implementation
```csharp
namespace Orleans.GpuBridge.Runtime.Memory;

public sealed class GpuMemoryPool : IGpuMemoryPool
{
    private readonly ILogger<GpuMemoryPool> _logger;
    private readonly IGpuBackendAdapter _adapter;
    private readonly int _deviceIndex;
    private readonly Stack<PooledBuffer>[] _pools;
    private readonly SemaphoreSlim _lock;
    private long _totalAllocated;
    private long _totalInUse;
    
    public GpuMemoryPool(
        ILogger<GpuMemoryPool> logger,
        IGpuBackendAdapter adapter,
        int deviceIndex)
    {
        _logger = logger;
        _adapter = adapter;
        _deviceIndex = deviceIndex;
        _lock = new SemaphoreSlim(1, 1);
        
        // Initialize pools for different size classes
        _pools = new Stack<PooledBuffer>[32]; // 2^0 to 2^31 bytes
        for (int i = 0; i < _pools.Length; i++)
        {
            _pools[i] = new Stack<PooledBuffer>();
        }
    }
    
    public async Task<IGpuBuffer> RentAsync(
        long sizeBytes,
        BufferUsage usage,
        CancellationToken ct)
    {
        var sizeClass = GetSizeClass(sizeBytes);
        var actualSize = 1L << sizeClass;
        
        await _lock.WaitAsync(ct);
        try
        {
            // Try to get from pool
            var pool = _pools[sizeClass];
            if (pool.TryPop(out var pooled))
            {
                pooled.Reset();
                Interlocked.Add(ref _totalInUse, actualSize);
                
                _logger.LogTrace(
                    "Rented {Size} bytes from pool (class {Class})",
                    actualSize, sizeClass);
                
                return pooled;
            }
            
            // Allocate new buffer
            var buffer = await _adapter.AllocateBufferAsync(
                _deviceIndex,
                actualSize,
                usage,
                ct);
            
            var pooledBuffer = new PooledBuffer(
                buffer,
                actualSize,
                this);
            
            Interlocked.Add(ref _totalAllocated, actualSize);
            Interlocked.Add(ref _totalInUse, actualSize);
            
            _logger.LogDebug(
                "Allocated new {Size} bytes buffer (class {Class})",
                actualSize, sizeClass);
            
            return pooledBuffer;
        }
        finally
        {
            _lock.Release();
        }
    }
    
    public async Task ReturnAsync(IGpuBuffer buffer)
    {
        if (buffer is not PooledBuffer pooled)
        {
            throw new ArgumentException("Buffer not from this pool");
        }
        
        var sizeClass = GetSizeClass(pooled.Size);
        
        await _lock.WaitAsync();
        try
        {
            var pool = _pools[sizeClass];
            
            // Return to pool if not too many
            if (pool.Count < 10) // Max 10 buffers per size class
            {
                pool.Push(pooled);
                Interlocked.Add(ref _totalInUse, -pooled.Size);
                
                _logger.LogTrace(
                    "Returned {Size} bytes to pool (class {Class})",
                    pooled.Size, sizeClass);
            }
            else
            {
                // Release buffer
                pooled.Dispose();
                Interlocked.Add(ref _totalAllocated, -pooled.Size);
                Interlocked.Add(ref _totalInUse, -pooled.Size);
                
                _logger.LogDebug(
                    "Released {Size} bytes buffer (pool full)",
                    pooled.Size);
            }
        }
        finally
        {
            _lock.Release();
        }
    }
    
    public MemoryPoolStats GetStats()
    {
        return new MemoryPoolStats(
            _totalAllocated,
            _totalInUse,
            _totalAllocated - _totalInUse,
            _pools.Sum(p => p.Count));
    }
    
    private static int GetSizeClass(long size)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive");
        
        // Find next power of 2
        var bits = 64 - BitOperations.LeadingZeroCount((ulong)(size - 1));
        return Math.Min(bits, 31);
    }
}

internal sealed class PooledBuffer : IGpuBuffer
{
    private readonly IGpuBuffer _underlying;
    private readonly GpuMemoryPool _pool;
    private bool _disposed;
    
    public long Size { get; }
    public BufferUsage Usage => _underlying.Usage;
    public IntPtr Handle => _underlying.Handle;
    
    public PooledBuffer(
        IGpuBuffer underlying,
        long size,
        GpuMemoryPool pool)
    {
        _underlying = underlying;
        Size = size;
        _pool = pool;
    }
    
    public ValueTask<Memory<byte>> MapAsync(
        MapMode mode,
        CancellationToken ct)
    {
        return _underlying.MapAsync(mode, ct);
    }
    
    public ValueTask UnmapAsync(CancellationToken ct)
    {
        return _underlying.UnmapAsync(ct);
    }
    
    public ValueTask CopyFromAsync(
        ReadOnlyMemory<byte> source,
        CancellationToken ct)
    {
        return _underlying.CopyFromAsync(source, ct);
    }
    
    public ValueTask CopyToAsync(
        Memory<byte> destination,
        CancellationToken ct)
    {
        return _underlying.CopyToAsync(destination, ct);
    }
    
    public void Reset()
    {
        _disposed = false;
    }
    
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _ = _pool.ReturnAsync(this);
        }
    }
}
```

## Week 6: Persistent Kernels & Ring Buffers

### Day 6-7: Persistent Kernel Host

#### PersistentKernelHost Implementation
```csharp
namespace Orleans.GpuBridge.Runtime;

public sealed class PersistentKernelHost : IPersistentKernelHost
{
    private readonly ILogger<PersistentKernelHost> _logger;
    private readonly IGpuBackendAdapter _adapter;
    private readonly Dictionary<KernelId, PersistentKernel> _kernels;
    private readonly CancellationTokenSource _shutdownCts;
    
    public PersistentKernelHost(
        ILogger<PersistentKernelHost> logger,
        IGpuBackendAdapter adapter)
    {
        _logger = logger;
        _adapter = adapter;
        _kernels = new Dictionary<KernelId, PersistentKernel>();
        _shutdownCts = new CancellationTokenSource();
    }
    
    public async Task StartAsync(CancellationToken ct)
    {
        _logger.LogInformation("Starting persistent kernel host");
        
        // Load kernel definitions
        var kernelDefs = await LoadKernelDefinitionsAsync();
        
        foreach (var def in kernelDefs)
        {
            if (def.IsPersistent)
            {
                var kernel = await CreatePersistentKernelAsync(def, ct);
                _kernels[def.Id] = kernel;
                
                // Start kernel execution loop
                _ = kernel.RunAsync(_shutdownCts.Token);
                
                _logger.LogInformation(
                    "Started persistent kernel: {Id}",
                    def.Id);
            }
        }
    }
    
    public async Task StopAsync(CancellationToken ct)
    {
        _logger.LogInformation("Stopping persistent kernel host");
        
        // Signal shutdown
        _shutdownCts.Cancel();
        
        // Wait for kernels to stop
        var stopTasks = _kernels.Values
            .Select(k => k.StopAsync(ct))
            .ToArray();
        
        await Task.WhenAll(stopTasks);
        
        _logger.LogInformation("All persistent kernels stopped");
    }
    
    public async Task<IKernelExecutor> GetExecutorAsync(
        KernelId kernelId,
        CancellationToken ct)
    {
        if (_kernels.TryGetValue(kernelId, out var kernel))
        {
            return kernel;
        }
        
        // Create on-demand kernel
        var def = await LoadKernelDefinitionAsync(kernelId);
        if (def.IsPersistent)
        {
            throw new InvalidOperationException(
                $"Kernel {kernelId} should be persistent but not running");
        }
        
        return new OnDemandKernel(_adapter, def);
    }
    
    private async Task<PersistentKernel> CreatePersistentKernelAsync(
        KernelDefinition def,
        CancellationToken ct)
    {
        // Compile kernel
        var compiled = await _adapter.CompileKernelAsync(
            def.Source,
            new KernelCompilationOptions
            {
                Name = def.Id.Value,
                TargetArch = def.TargetArchitecture,
                OptLevel = OptimizationLevel.O3
            },
            ct);
        
        // Create ring buffers
        var inputRing = new RingBuffer(
            def.InputBufferSize,
            _adapter,
            def.PreferredDevice);
        var outputRing = new RingBuffer(
            def.OutputBufferSize,
            _adapter,
            def.PreferredDevice);
        
        await inputRing.InitializeAsync(ct);
        await outputRing.InitializeAsync(ct);
        
        return new PersistentKernel(
            def,
            compiled,
            inputRing,
            outputRing,
            _adapter,
            _logger);
    }
}

internal sealed class PersistentKernel : IKernelExecutor
{
    private readonly KernelDefinition _definition;
    private readonly ICompiledKernel _compiled;
    private readonly RingBuffer _inputRing;
    private readonly RingBuffer _outputRing;
    private readonly IGpuBackendAdapter _adapter;
    private readonly ILogger _logger;
    private readonly Channel<WorkRequest> _requests;
    
    public PersistentKernel(
        KernelDefinition definition,
        ICompiledKernel compiled,
        RingBuffer inputRing,
        RingBuffer outputRing,
        IGpuBackendAdapter adapter,
        ILogger logger)
    {
        _definition = definition;
        _compiled = compiled;
        _inputRing = inputRing;
        _outputRing = outputRing;
        _adapter = adapter;
        _logger = logger;
        _requests = Channel.CreateUnbounded<WorkRequest>();
    }
    
    public async Task<KernelResult> ExecuteAsync(
        ReadOnlyMemory<byte> input,
        GpuExecutionHints? hints,
        CancellationToken ct)
    {
        var request = new WorkRequest(
            Guid.NewGuid(),
            input,
            new TaskCompletionSource<KernelResult>());
        
        await _requests.Writer.WriteAsync(request, ct);
        
        return await request.Completion.Task;
    }
    
    public async Task RunAsync(CancellationToken ct)
    {
        _logger.LogInformation(
            "Persistent kernel {Id} starting",
            _definition.Id);
        
        try
        {
            // Launch GPU kernel
            var launchParams = new KernelLaunchParameters
            {
                GlobalWorkSize = _definition.GlobalWorkSize,
                LocalWorkSize = _definition.LocalWorkSize,
                Arguments = new[]
                {
                    _inputRing.DeviceBuffer,
                    _outputRing.DeviceBuffer,
                    _inputRing.GetControlBuffer(),
                    _outputRing.GetControlBuffer()
                }
            };
            
            var handle = await _adapter.LaunchKernelAsync(
                _compiled,
                launchParams,
                ct);
            
            // Process requests
            await ProcessRequestsAsync(ct);
            
            // Wait for kernel completion
            await handle.WaitAsync(ct);
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation(
                "Persistent kernel {Id} stopping",
                _definition.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex,
                "Persistent kernel {Id} failed",
                _definition.Id);
        }
    }
    
    private async Task ProcessRequestsAsync(CancellationToken ct)
    {
        await foreach (var request in _requests.Reader.ReadAllAsync(ct))
        {
            try
            {
                // Write to input ring
                var inputOffset = await _inputRing.WriteAsync(
                    request.Input,
                    ct);
                
                // Wait for processing
                await Task.Delay(10, ct); // Polling interval
                
                // Read from output ring
                var output = await _outputRing.ReadAsync(
                    _definition.OutputSize,
                    ct);
                
                request.Completion.SetResult(
                    new KernelResult(output, TimeSpan.Zero));
            }
            catch (Exception ex)
            {
                request.Completion.SetException(ex);
            }
        }
    }
    
    public Task StopAsync(CancellationToken ct)
    {
        _requests.Writer.TryComplete();
        return Task.CompletedTask;
    }
    
    private record WorkRequest(
        Guid Id,
        ReadOnlyMemory<byte> Input,
        TaskCompletionSource<KernelResult> Completion);
}
```

### Day 8-9: Ring Buffer Implementation

#### Zero-Copy Ring Buffer
```csharp
namespace Orleans.GpuBridge.Runtime.Memory;

public sealed class RingBuffer
{
    private readonly long _size;
    private readonly IGpuBackendAdapter _adapter;
    private readonly int _deviceIndex;
    private IGpuBuffer _deviceBuffer = default!;
    private Memory<byte> _hostMemory;
    private readonly SemaphoreSlim _writeLock;
    private readonly SemaphoreSlim _readLock;
    private long _writePosition;
    private long _readPosition;
    private long _bytesAvailable;
    
    public IGpuBuffer DeviceBuffer => _deviceBuffer;
    
    public RingBuffer(
        long size,
        IGpuBackendAdapter adapter,
        int deviceIndex)
    {
        _size = size;
        _adapter = adapter;
        _deviceIndex = deviceIndex;
        _writeLock = new SemaphoreSlim(1, 1);
        _readLock = new SemaphoreSlim(1, 1);
    }
    
    public async Task InitializeAsync(CancellationToken ct)
    {
        // Allocate unified memory (accessible from both CPU and GPU)
        _deviceBuffer = await _adapter.AllocateBufferAsync(
            _deviceIndex,
            _size,
            BufferUsage.UnifiedMemory | BufferUsage.Persistent,
            ct);
        
        // Map for host access
        _hostMemory = await _deviceBuffer.MapAsync(
            MapMode.ReadWrite | MapMode.Persistent,
            ct);
    }
    
    public async Task<long> WriteAsync(
        ReadOnlyMemory<byte> data,
        CancellationToken ct)
    {
        await _writeLock.WaitAsync(ct);
        try
        {
            var dataLength = data.Length;
            
            // Check available space
            var available = _size - _bytesAvailable;
            if (dataLength > available)
            {
                throw new InsufficientBufferSpaceException(
                    $"Need {dataLength} bytes, only {available} available");
            }
            
            // Calculate write segments
            var firstSegmentSize = Math.Min(
                dataLength,
                _size - _writePosition);
            var secondSegmentSize = dataLength - firstSegmentSize;
            
            // Write first segment
            data.Slice(0, firstSegmentSize).CopyTo(
                _hostMemory.Slice((int)_writePosition, firstSegmentSize));
            
            // Write second segment (wrap around)
            if (secondSegmentSize > 0)
            {
                data.Slice(firstSegmentSize, secondSegmentSize).CopyTo(
                    _hostMemory.Slice(0, secondSegmentSize));
            }
            
            // Update position
            var oldPosition = _writePosition;
            _writePosition = (_writePosition + dataLength) % _size;
            Interlocked.Add(ref _bytesAvailable, dataLength);
            
            // Memory barrier to ensure visibility to GPU
            Thread.MemoryBarrier();
            
            return oldPosition;
        }
        finally
        {
            _writeLock.Release();
        }
    }
    
    public async Task<Memory<byte>> ReadAsync(
        int length,
        CancellationToken ct)
    {
        await _readLock.WaitAsync(ct);
        try
        {
            // Wait for data
            while (_bytesAvailable < length)
            {
                await Task.Delay(1, ct);
            }
            
            // Allocate result buffer
            var result = new byte[length];
            var resultMemory = new Memory<byte>(result);
            
            // Calculate read segments
            var firstSegmentSize = Math.Min(
                length,
                _size - _readPosition);
            var secondSegmentSize = length - firstSegmentSize;
            
            // Read first segment
            _hostMemory.Slice((int)_readPosition, firstSegmentSize)
                .CopyTo(resultMemory.Slice(0, firstSegmentSize));
            
            // Read second segment (wrap around)
            if (secondSegmentSize > 0)
            {
                _hostMemory.Slice(0, secondSegmentSize)
                    .CopyTo(resultMemory.Slice(firstSegmentSize));
            }
            
            // Update position
            _readPosition = (_readPosition + length) % _size;
            Interlocked.Add(ref _bytesAvailable, -length);
            
            return resultMemory;
        }
        finally
        {
            _readLock.Release();
        }
    }
    
    public IGpuBuffer GetControlBuffer()
    {
        // Control buffer contains read/write positions for GPU access
        var control = new RingBufferControl
        {
            Size = _size,
            WritePosition = Interlocked.Read(ref _writePosition),
            ReadPosition = Interlocked.Read(ref _readPosition),
            BytesAvailable = Interlocked.Read(ref _bytesAvailable)
        };
        
        // This would be a small unified buffer updated atomically
        // For now, return placeholder
        return _deviceBuffer;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct RingBufferControl
{
    public long Size;
    public long WritePosition;
    public long ReadPosition;
    public long BytesAvailable;
}
```

### Day 10: Hardware Testing & Benchmarks

#### GPU Hardware Tests
```csharp
[Category("GPU")]
[Collection("HardwareTests")]
public class GpuRuntimeTests : IAsyncLifetime
{
    private IGpuBackendAdapter _adapter = default!;
    private IServiceProvider _services = default!;
    
    public async Task InitializeAsync()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddSingleton<IGpuBackendAdapter, DotComputeAdapter>();
        services.Configure<DotComputeOptions>(o =>
        {
            o.PreferredBackend = GpuBackend.Cuda;
            o.EnableProfiling = true;
        });
        
        _services = services.BuildServiceProvider();
        _adapter = _services.GetRequiredService<IGpuBackendAdapter>();
        
        await _adapter.InitializeAsync(CancellationToken.None);
    }
    
    public async Task DisposeAsync()
    {
        await _adapter.ShutdownAsync(CancellationToken.None);
    }
    
    [SkippableFact]
    public async Task Should_Execute_Vector_Add_On_GPU()
    {
        Skip.IfNot(await HasGpuAsync(), "No GPU available");
        
        const int size = 1024 * 1024;
        var a = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
        var b = Enumerable.Range(0, size).Select(i => (float)i * 2).ToArray();
        
        var kernel = await CompileVectorAddKernelAsync();
        
        var inputBuffer = await AllocateAndFillAsync(a, b);
        var outputBuffer = await _adapter.AllocateBufferAsync(
            0, size * sizeof(float), BufferUsage.WriteOnly);
        
        var handle = await _adapter.LaunchKernelAsync(
            kernel,
            new KernelLaunchParameters
            {
                GlobalWorkSize = size,
                LocalWorkSize = 256,
                Arguments = new[] { inputBuffer, outputBuffer }
            });
        
        await handle.WaitAsync();
        
        var result = new float[size];
        await outputBuffer.CopyToAsync(
            MemoryMarshal.AsBytes<float>(result));
        
        // Verify results
        for (int i = 0; i < size; i++)
        {
            Assert.Equal(a[i] + b[i], result[i], 0.001f);
        }
    }
    
    [SkippableFact]
    public async Task Persistent_Kernel_Should_Process_Stream()
    {
        Skip.IfNot(await HasGpuAsync(), "No GPU available");
        
        var host = _services.GetRequiredService<IPersistentKernelHost>();
        await host.StartAsync(CancellationToken.None);
        
        try
        {
            var executor = await host.GetExecutorAsync(
                new KernelId("test/persistent"));
            
            const int iterations = 100;
            var latencies = new List<double>();
            
            for (int i = 0; i < iterations; i++)
            {
                var input = BitConverter.GetBytes(i);
                
                var sw = Stopwatch.StartNew();
                var result = await executor.ExecuteAsync(
                    input,
                    null,
                    CancellationToken.None);
                sw.Stop();
                
                latencies.Add(sw.ElapsedMilliseconds);
                
                var output = BitConverter.ToInt32(result.Data.Span);
                Assert.Equal(i * 2, output); // Kernel doubles the input
            }
            
            var avgLatency = latencies.Average();
            var p50 = latencies.OrderBy(l => l).Skip(50).First();
            var p95 = latencies.OrderBy(l => l).Skip(95).First();
            
            _output.WriteLine($"Latencies - Avg: {avgLatency:F2}ms, " +
                             $"P50: {p50:F2}ms, P95: {p95:F2}ms");
            
            Assert.True(p50 < 1.0, "P50 latency should be sub-millisecond");
        }
        finally
        {
            await host.StopAsync(CancellationToken.None);
        }
    }
    
    private async Task<bool> HasGpuAsync()
    {
        var devices = await _adapter.GetDevicesAsync();
        return devices.Any(d => d.Type == DeviceType.Gpu);
    }
}
```

## Deliverables Checklist

### Week 5 Deliverables
- [ ] DotCompute adapter implementation
- [ ] Device broker with queue management
- [ ] GPU memory pool with size classes
- [ ] Resource allocation and tracking
- [ ] Basic kernel compilation
- [ ] Unit tests for memory management

### Week 6 Deliverables
- [ ] Persistent kernel host
- [ ] Ring buffer implementation
- [ ] Zero-copy memory mapping
- [ ] Kernel execution loop
- [ ] Hardware-dependent tests
- [ ] Performance benchmarks

## Performance Targets

### Kernel Execution
- Kernel launch overhead < 100μs
- Persistent kernel latency < 1ms (P50)
- Memory transfer bandwidth > 10GB/s
- Queue processing > 10K ops/sec

### Memory Management
- Buffer allocation < 1ms
- Pool hit ratio > 90%
- Zero-copy for buffers > 64KB
- Memory fragmentation < 10%

### Resource Utilization
- GPU utilization > 80% under load
- Memory efficiency > 85%
- Power efficiency within thermal limits

## Next Phase Preview

Phase 4 will focus on production hardening:
- CUDA Graph optimization
- GPUDirect Storage integration
- Comprehensive telemetry
- Performance tuning
- Production samples and documentation