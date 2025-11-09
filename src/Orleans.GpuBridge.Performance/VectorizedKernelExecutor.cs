using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Ultra-high-performance vectorized kernel executor with CPU-specific optimizations
/// </summary>
public sealed partial class VectorizedKernelExecutor : IDisposable
{
    private readonly ILogger<VectorizedKernelExecutor> _logger;
    private readonly HighPerformanceMemoryPool<float> _memoryPool;
    private readonly int _coreCount;
    private readonly bool _hasAvx512;
    private readonly bool _hasAvx2;
    private readonly bool _hasAvx;
    private readonly bool _hasFma;
    private readonly bool _hasNeon;
    private readonly Channel<WorkItem> _workChannel;
    private readonly Task[] _workerTasks;
    private readonly CancellationTokenSource _cancellationSource;
    private volatile bool _disposed;

    // CPU detection results
    public bool HasAvx512 => _hasAvx512;
    public bool HasAvx2 => _hasAvx2;
    public bool HasAvx => _hasAvx;
    public bool HasFma => _hasFma;
    public bool HasNeon => _hasNeon;

    public VectorizedKernelExecutor(
        ILogger<VectorizedKernelExecutor> logger,
        HighPerformanceMemoryPool<float> memoryPool,
        int? workerThreads = null)
    {
        _logger = logger;
        _memoryPool = memoryPool;
        _coreCount = Environment.ProcessorCount;

        // Detect CPU capabilities
        _hasAvx512 = Avx512F.IsSupported;
        _hasAvx2 = Avx2.IsSupported;
        _hasAvx = Avx.IsSupported;
        _hasFma = Fma.IsSupported;
        _hasNeon = AdvSimd.IsSupported;

        var threadCount = workerThreads ?? Math.Min(_coreCount, 8);
        _workChannel = Channel.CreateUnbounded<WorkItem>();
        _cancellationSource = new CancellationTokenSource();

        // Start worker threads
        _workerTasks = new Task[threadCount];
        for (int i = 0; i < threadCount; i++)
        {
            var threadId = i;
            _workerTasks[i] = Task.Run(() => WorkerLoop(threadId, _cancellationSource.Token));
        }

        _logger.LogInformation(
            "Vectorized executor initialized: Cores={Cores}, Workers={Workers}, AVX512={Avx512}, AVX2={Avx2}, FMA={Fma}, NEON={Neon}",
            _coreCount, threadCount, _hasAvx512, _hasAvx2, _hasFma, _hasNeon);
    }

    /// <summary>
    /// Execute vector addition with maximum SIMD utilization
    /// </summary>
    public async Task<float[]> VectorAddAsync(float[] a, float[] b, CancellationToken ct = default)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new float[a.Length];
        var workItem = new VectorAddWorkItem(a, b, result);

        await ExecuteWorkItemAsync(workItem, ct);
        return result;
    }

    /// <summary>
    /// Execute fused multiply-add with optimal vectorization
    /// </summary>
    public async Task<float[]> FusedMultiplyAddAsync(float[] a, float[] b, float[] c, CancellationToken ct = default)
    {
        if (a.Length != b.Length || b.Length != c.Length)
            throw new ArgumentException("Vector lengths must match");

        var result = new float[a.Length];
        var workItem = new FmaWorkItem(a, b, c, result);

        await ExecuteWorkItemAsync(workItem, ct);
        return result;
    }

    /// <summary>
    /// Execute matrix multiplication with cache-optimized blocking
    /// </summary>
    public async Task<float[]> MatrixMultiplyAsync(float[] a, float[] b, int m, int n, int k, CancellationToken ct = default)
    {
        var result = new float[m * n];
        var workItem = new MatrixMultiplyWorkItem(a, b, result, m, n, k);

        await ExecuteWorkItemAsync(workItem, ct);
        return result;
    }

    /// <summary>
    /// Execute reduction operation (sum, min, max)
    /// </summary>
    public async Task<float> ReduceAsync(float[] input, ReductionOperation operation, CancellationToken ct = default)
    {
        var result = new float[1];
        var workItem = new ReductionWorkItem(input, result, operation);

        await ExecuteWorkItemAsync(workItem, ct);
        return result[0];
    }

    private async Task ExecuteWorkItemAsync(WorkItem workItem, CancellationToken ct)
    {
        var tcs = new TaskCompletionSource<bool>();
        workItem.CompletionSource = tcs;

        await _workChannel.Writer.WriteAsync(workItem, ct);
        await tcs.Task;
    }

    private async Task WorkerLoop(int threadId, CancellationToken ct)
    {
        _logger.LogDebug("Worker thread {ThreadId} started", threadId);

        // Set thread affinity for better cache locality
        SetThreadAffinity(threadId);

        try
        {
            await foreach (var workItem in _workChannel.Reader.ReadAllAsync(ct))
            {
                try
                {
                    ExecuteWorkItem(workItem);
                    workItem.CompletionSource.SetResult(true);
                }
                catch (Exception ex)
                {
                    workItem.CompletionSource.SetException(ex);
                }
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // Expected cancellation
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Worker thread {ThreadId} failed", threadId);
        }

        _logger.LogDebug("Worker thread {ThreadId} stopped", threadId);
    }

    private void ExecuteWorkItem(WorkItem workItem)
    {
        switch (workItem)
        {
            case VectorAddWorkItem add:
                ExecuteVectorAdd(add.A, add.B, add.Result);
                break;
            case FmaWorkItem fma:
                ExecuteFusedMultiplyAdd(fma.A, fma.B, fma.C, fma.Result);
                break;
            case MatrixMultiplyWorkItem mm:
                ExecuteMatrixMultiply(mm.A, mm.B, mm.Result, mm.M, mm.N, mm.K);
                break;
            case ReductionWorkItem red:
                ExecuteReduction(red.Input, red.Result, red.Operation);
                break;
        }
    }

    private void SetThreadAffinity(int threadId)
    {
        if (!OperatingSystem.IsWindows()) return;

        try
        {
            var affinityMask = (UIntPtr)(1UL << (threadId % _coreCount));
            SetThreadAffinityMask(GetCurrentThread(), affinityMask);
        }
        catch
        {
            // Ignore affinity errors
        }
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();

    [DllImport("kernel32.dll")]
    private static extern UIntPtr SetThreadAffinityMask(IntPtr hThread, UIntPtr dwThreadAffinityMask);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _logger.LogDebug("Disposing vectorized kernel executor");

        _workChannel.Writer.Complete();
        _cancellationSource.Cancel();

        try
        {
            Task.WaitAll(_workerTasks, TimeSpan.FromSeconds(5));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error waiting for worker tasks to complete");
        }

        _cancellationSource.Dispose();
    }
}
