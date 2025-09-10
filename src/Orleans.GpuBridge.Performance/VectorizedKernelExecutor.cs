using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Ultra-high-performance vectorized kernel executor with CPU-specific optimizations
/// </summary>
public sealed class VectorizedKernelExecutor : IDisposable
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAdd(float[] a, float[] b, float[] result)
    {
        var length = a.Length;
        
        if (_hasAvx512 && length >= Vector512<float>.Count)
        {
            ExecuteVectorAddAvx512(a, b, result);
        }
        else if (_hasAvx2 && length >= Vector256<float>.Count)
        {
            ExecuteVectorAddAvx2(a, b, result);
        }
        else if (_hasNeon && length >= Vector128<float>.Count)
        {
            ExecuteVectorAddNeon(a, b, result);
        }
        else
        {
            ExecuteVectorAddGeneric(a, b, result);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAddAvx512(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector512<float>.Count;
        var vectorCount = a.Length / vectorSize;
        
        fixed (float* pA = a, pB = b, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = Avx512F.LoadVector512(pA + offset);
                var vecB = Avx512F.LoadVector512(pB + offset);
                var vecResult = Avx512F.Add(vecA, vecB);
                Avx512F.Store(pResult + offset, vecResult);
            }
            
            // Handle remainder
            var remaining = a.Length % vectorSize;
            if (remaining > 0)
            {
                var startIdx = vectorCount * vectorSize;
                for (int i = 0; i < remaining; i++)
                {
                    result[startIdx + i] = a[startIdx + i] + b[startIdx + i];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAddAvx2(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector256<float>.Count;
        var vectorCount = a.Length / vectorSize;
        
        fixed (float* pA = a, pB = b, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = Avx.LoadVector256(pA + offset);
                var vecB = Avx.LoadVector256(pB + offset);
                var vecResult = Avx.Add(vecA, vecB);
                Avx.Store(pResult + offset, vecResult);
            }
        }
        
        // Handle remainder with scalar operations
        HandleVectorRemainder(a, b, result, vectorCount * vectorSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteVectorAddNeon(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector128<float>.Count;
        var vectorCount = a.Length / vectorSize;
        
        fixed (float* pA = a, pB = b, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = AdvSimd.LoadVector128(pA + offset);
                var vecB = AdvSimd.LoadVector128(pB + offset);
                var vecResult = AdvSimd.Add(vecA, vecB);
                AdvSimd.Store(pResult + offset, vecResult);
            }
        }
        
        HandleVectorRemainder(a, b, result, vectorCount * vectorSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteVectorAddGeneric(float[] a, float[] b, float[] result)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = a.Length / vectorSize;
        
        for (int i = 0; i < vectorCount; i++)
        {
            var offset = i * vectorSize;
            var vecA = new Vector<float>(a, offset);
            var vecB = new Vector<float>(b, offset);
            var vecResult = vecA + vecB;
            vecResult.CopyTo(result, offset);
        }
        
        HandleVectorRemainder(a, b, result, vectorCount * vectorSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteFusedMultiplyAdd(float[] a, float[] b, float[] c, float[] result)
    {
        var length = a.Length;
        
        if (_hasFma && _hasAvx2 && length >= Vector256<float>.Count)
        {
            ExecuteFmaAvx2(a, b, c, result);
        }
        else if (_hasNeon && length >= Vector128<float>.Count)
        {
            ExecuteFmaNeon(a, b, c, result);
        }
        else
        {
            ExecuteFmaGeneric(a, b, c, result);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteFmaAvx2(float[] a, float[] b, float[] c, float[] result)
    {
        var vectorSize = Vector256<float>.Count;
        var vectorCount = a.Length / vectorSize;
        
        fixed (float* pA = a, pB = b, pC = c, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = Avx.LoadVector256(pA + offset);
                var vecB = Avx.LoadVector256(pB + offset);
                var vecC = Avx.LoadVector256(pC + offset);
                var vecResult = Fma.MultiplyAdd(vecA, vecB, vecC);
                Avx.Store(pResult + offset, vecResult);
            }
        }
        
        // Handle remainder
        var startIdx = vectorCount * vectorSize;
        for (int i = startIdx; i < a.Length; i++)
        {
            result[i] = a[i] * b[i] + c[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe void ExecuteFmaNeon(float[] a, float[] b, float[] c, float[] result)
    {
        var vectorSize = Vector128<float>.Count;
        var vectorCount = a.Length / vectorSize;
        
        fixed (float* pA = a, pB = b, pC = c, pResult = result)
        {
            for (int i = 0; i < vectorCount; i++)
            {
                var offset = i * vectorSize;
                var vecA = AdvSimd.LoadVector128(pA + offset);
                var vecB = AdvSimd.LoadVector128(pB + offset);
                var vecC = AdvSimd.LoadVector128(pC + offset);
                var vecResult = AdvSimd.FusedMultiplyAdd(vecC, vecA, vecB);
                AdvSimd.Store(pResult + offset, vecResult);
            }
        }
        
        // Handle remainder
        var startIdx = vectorCount * vectorSize;
        for (int i = startIdx; i < a.Length; i++)
        {
            result[i] = a[i] * b[i] + c[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteFmaGeneric(float[] a, float[] b, float[] c, float[] result)
    {
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = Math.FusedMultiplyAdd(a[i], b[i], c[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteMatrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k)
    {
        // Cache-optimal blocked matrix multiplication
        const int blockSize = 64; // Optimized for L1 cache
        
        Parallel.For(0, (m + blockSize - 1) / blockSize, i =>
        {
            var iStart = i * blockSize;
            var iEnd = Math.Min(iStart + blockSize, m);
            
            for (int j = 0; j < n; j += blockSize)
            {
                var jEnd = Math.Min(j + blockSize, n);
                
                for (int kBlock = 0; kBlock < k; kBlock += blockSize)
                {
                    var kEnd = Math.Min(kBlock + blockSize, k);
                    
                    ExecuteMatrixBlock(a, b, result, iStart, iEnd, j, jEnd, kBlock, kEnd, m, n, k);
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteMatrixBlock(float[] a, float[] b, float[] result, 
        int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
        int m, int n, int k)
    {
        for (int i = iStart; i < iEnd; i++)
        {
            for (int j = jStart; j < jEnd; j++)
            {
                var sum = 0.0f;
                var resultIndex = i * n + j;
                
                // Vectorized inner loop
                var kVectorized = (kEnd - kStart) / Vector<float>.Count * Vector<float>.Count;
                var sumVector = Vector<float>.Zero;
                
                int kIdx = kStart;
                for (; kIdx < kStart + kVectorized; kIdx += Vector<float>.Count)
                {
                    var aVector = new Vector<float>(a, i * k + kIdx);
                    var bVector = LoadBVector(b, kIdx, j, n, Vector<float>.Count);
                    sumVector += aVector * bVector;
                }
                
                // Sum vector components
                for (int v = 0; v < Vector<float>.Count; v++)
                {
                    sum += sumVector[v];
                }
                
                // Handle remaining elements
                for (; kIdx < kEnd; kIdx++)
                {
                    sum += a[i * k + kIdx] * b[kIdx * n + j];
                }
                
                result[resultIndex] += sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private Vector<float> LoadBVector(float[] b, int kStart, int j, int n, int count)
    {
        // Load non-contiguous B matrix elements into vector
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = b[(kStart + i) * n + j];
        }
        return new Vector<float>(values);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteReduction(float[] input, float[] result, ReductionOperation operation)
    {
        var length = input.Length;
        if (length == 0)
        {
            result[0] = 0;
            return;
        }

        switch (operation)
        {
            case ReductionOperation.Sum:
                result[0] = VectorizedSum(input);
                break;
            case ReductionOperation.Max:
                result[0] = VectorizedMax(input);
                break;
            case ReductionOperation.Min:
                result[0] = VectorizedMin(input);
                break;
            default:
                throw new ArgumentException($"Unsupported reduction operation: {operation}");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float VectorizedSum(float[] input)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var sumVector = Vector<float>.Zero;
        
        // Vectorized sum
        for (int i = 0; i < vectorCount; i++)
        {
            var vec = new Vector<float>(input, i * vectorSize);
            sumVector += vec;
        }
        
        // Sum vector components
        float sum = 0;
        for (int i = 0; i < vectorSize; i++)
        {
            sum += sumVector[i];
        }
        
        // Add remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            sum += input[i];
        }
        
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float VectorizedMax(float[] input)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var maxVector = new Vector<float>(float.MinValue);
        
        // Vectorized max
        for (int i = 0; i < vectorCount; i++)
        {
            var vec = new Vector<float>(input, i * vectorSize);
            maxVector = Vector.Max(maxVector, vec);
        }
        
        // Find max in vector
        float max = float.MinValue;
        for (int i = 0; i < vectorSize; i++)
        {
            max = Math.Max(max, maxVector[i]);
        }
        
        // Check remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            max = Math.Max(max, input[i]);
        }
        
        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float VectorizedMin(float[] input)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var minVector = new Vector<float>(float.MaxValue);
        
        // Vectorized min
        for (int i = 0; i < vectorCount; i++)
        {
            var vec = new Vector<float>(input, i * vectorSize);
            minVector = Vector.Min(minVector, vec);
        }
        
        // Find min in vector
        float min = float.MaxValue;
        for (int i = 0; i < vectorSize; i++)
        {
            min = Math.Min(min, minVector[i]);
        }
        
        // Check remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            min = Math.Min(min, input[i]);
        }
        
        return min;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void HandleVectorRemainder(float[] a, float[] b, float[] result, int startIndex)
    {
        for (int i = startIndex; i < a.Length; i++)
        {
            result[i] = a[i] + b[i];
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

    // Work item types
    private abstract record WorkItem
    {
        public TaskCompletionSource<bool> CompletionSource { get; set; } = null!;
    }

    private record VectorAddWorkItem(float[] A, float[] B, float[] Result) : WorkItem;
    private record FmaWorkItem(float[] A, float[] B, float[] C, float[] Result) : WorkItem;
    private record MatrixMultiplyWorkItem(float[] A, float[] B, float[] Result, int M, int N, int K) : WorkItem;
    private record ReductionWorkItem(float[] Input, float[] Result, ReductionOperation Operation) : WorkItem;
}

public enum ReductionOperation
{
    Sum,
    Max,
    Min
}