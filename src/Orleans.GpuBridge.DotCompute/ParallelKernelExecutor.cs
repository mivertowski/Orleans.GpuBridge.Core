using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.DotCompute;

/// <summary>
/// High-performance parallel kernel executor with SIMD support
/// </summary>
public sealed class ParallelKernelExecutor
{
    private readonly ILogger<ParallelKernelExecutor> _logger;
    private readonly int _maxParallelism;
    private readonly bool _useSimd;
    
    public ParallelKernelExecutor(ILogger<ParallelKernelExecutor> logger)
    {
        _logger = logger;
        _maxParallelism = Environment.ProcessorCount;
        _useSimd = DetectSimdSupport();
        
        _logger.LogInformation(
            "Parallel executor initialized with {Cores} cores, SIMD: {Simd}",
            _maxParallelism, _useSimd);
    }
    
    private bool DetectSimdSupport()
    {
        if (Avx512F.IsSupported)
        {
            _logger.LogDebug("AVX-512 support detected");
            return true;
        }
        if (Avx2.IsSupported)
        {
            _logger.LogDebug("AVX2 support detected");
            return true;
        }
        if (Avx.IsSupported)
        {
            _logger.LogDebug("AVX support detected");
            return true;
        }
        if (AdvSimd.IsSupported)
        {
            _logger.LogDebug("ARM NEON support detected");
            return true;
        }
        
        return false;
    }
    
    /// <summary>
    /// Executes a compute kernel in parallel with optimal work distribution
    /// </summary>
    public async Task<TOut[]> ExecuteAsync<TIn, TOut>(
        TIn[] input,
        Func<TIn, TOut> kernel,
        ParallelExecutionOptions? options = null,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull
    {
        options ??= new ParallelExecutionOptions();
        
        var inputCount = input.Length;
        var output = new TOut[inputCount];
        
        if (inputCount == 0)
            return output;
        
        // Determine optimal chunk size
        var chunkSize = CalculateOptimalChunkSize(inputCount, options);
        var chunks = CreateWorkChunks(inputCount, chunkSize);
        
        _logger.LogDebug(
            "Executing kernel on {Count} items with {Chunks} chunks of size ~{Size}",
            inputCount, chunks.Count, chunkSize);
        
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Execute chunks in parallel
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = options.MaxDegreeOfParallelism ?? _maxParallelism,
            CancellationToken = ct
        };
        
        await Parallel.ForEachAsync(chunks, parallelOptions, async (chunk, token) =>
        {
            await Task.Run(() => ProcessChunk(input, output, kernel, chunk, options), token);
        });
        
        stopwatch.Stop();
        
        _logger.LogInformation(
            "Executed kernel on {Count} items in {Time}ms ({Rate:N0} items/sec)",
            inputCount,
            stopwatch.ElapsedMilliseconds,
            inputCount / stopwatch.Elapsed.TotalSeconds);
        
        return output;
    }
    
    /// <summary>
    /// Executes a vectorized kernel for numerical operations
    /// </summary>
    public async Task<float[]> ExecuteVectorizedAsync(
        float[] input,
        VectorOperation operation,
        float[] parameters,
        CancellationToken ct = default)
    {
        var output = new float[input.Length];
        
        if (!_useSimd || input.Length < Vector<float>.Count * 2)
        {
            // Fallback to scalar operations
            await Task.Run(() => ExecuteScalar(input, output, operation, parameters), ct);
            return output;
        }
        
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Process vectorized portions
        await Task.Run(() =>
        {
            if (Avx2.IsSupported)
            {
                ExecuteAvx2(input, output, operation, parameters);
            }
            else if (Avx.IsSupported)
            {
                ExecuteAvx(input, output, operation, parameters);
            }
            else if (AdvSimd.IsSupported)
            {
                ExecuteNeon(input, output, operation, parameters);
            }
            else
            {
                ExecuteGenericVector(input, output, operation, parameters);
            }
        }, ct);
        
        stopwatch.Stop();
        
        var throughput = (input.Length * sizeof(float)) / (stopwatch.Elapsed.TotalSeconds * 1024 * 1024);
        _logger.LogDebug(
            "Vectorized operation completed: {Length} floats in {Time}ms ({Throughput:N0} MB/s)",
            input.Length, stopwatch.ElapsedMilliseconds, throughput);
        
        return output;
    }
    
    private void ExecuteAvx2(
        float[] input,
        float[] output,
        VectorOperation operation,
        float[] parameters)
    {
        var vectorSize = Vector256<float>.Count;
        var vectorCount = input.Length / vectorSize;
        var remainder = input.Length % vectorSize;
        
        // Process vectors
        Parallel.For(0, vectorCount, i =>
        {
            unsafe
            {
                fixed (float* pInput = input)
                fixed (float* pOutput = output)
                {
                    var offset = i * vectorSize;
                    var vec = Avx.LoadVector256(pInput + offset);
                    
                    var result = operation switch
                    {
                        VectorOperation.Add => Avx.Add(vec, Vector256.Create(parameters[0])),
                        VectorOperation.Multiply => Avx.Multiply(vec, Vector256.Create(parameters[0])),
                        VectorOperation.FusedMultiplyAdd => Fma.MultiplyAdd(
                            vec,
                            Vector256.Create(parameters[0]),
                            Vector256.Create(parameters[1])),
                        VectorOperation.Sqrt => Avx.Sqrt(vec),
                        VectorOperation.Reciprocal => Avx.Reciprocal(vec),
                        VectorOperation.Max => Avx.Max(vec, Vector256.Create(parameters[0])),
                        VectorOperation.Min => Avx.Min(vec, Vector256.Create(parameters[0])),
                        _ => vec
                    };
                    
                    Avx.Store(pOutput + offset, result);
                }
            }
        });
        
        // Process remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            output[i] = ApplyScalarOperation(input[i], operation, parameters);
        }
    }
    
    private void ExecuteAvx(
        float[] input,
        float[] output,
        VectorOperation operation,
        float[] parameters)
    {
        // Similar to AVX2 but using 256-bit vectors
        ExecuteAvx2(input, output, operation, parameters);
    }
    
    private void ExecuteNeon(
        float[] input,
        float[] output,
        VectorOperation operation,
        float[] parameters)
    {
        var vectorSize = Vector128<float>.Count;
        var vectorCount = input.Length / vectorSize;
        
        Parallel.For(0, vectorCount, i =>
        {
            unsafe
            {
                fixed (float* pInput = input)
                fixed (float* pOutput = output)
                {
                    var offset = i * vectorSize;
                    var vec = AdvSimd.LoadVector128(pInput + offset);
                    
                    var result = operation switch
                    {
                        VectorOperation.Add => AdvSimd.Add(vec, Vector128.Create(parameters[0])),
                        VectorOperation.Multiply => AdvSimd.Multiply(vec, Vector128.Create(parameters[0])),
                        VectorOperation.FusedMultiplyAdd => AdvSimd.FusedMultiplyAdd(
                            Vector128.Create(parameters[1]),
                            vec,
                            Vector128.Create(parameters[0])),
                        VectorOperation.Sqrt => AdvSimd.Arm64.Sqrt(vec),
                        VectorOperation.Reciprocal => AdvSimd.ReciprocalEstimate(vec),
                        VectorOperation.Max => AdvSimd.Max(vec, Vector128.Create(parameters[0])),
                        VectorOperation.Min => AdvSimd.Min(vec, Vector128.Create(parameters[0])),
                        _ => vec
                    };
                    
                    AdvSimd.Store(pOutput + offset, result);
                }
            }
        });
        
        // Process remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            output[i] = ApplyScalarOperation(input[i], operation, parameters);
        }
    }
    
    private void ExecuteGenericVector(
        float[] input,
        float[] output,
        VectorOperation operation,
        float[] parameters)
    {
        var vectorSize = Vector<float>.Count;
        var vectorCount = input.Length / vectorSize;
        
        Parallel.For(0, vectorCount, i =>
        {
            var offset = i * vectorSize;
            var vec = new Vector<float>(input, offset);
            
            var result = operation switch
            {
                VectorOperation.Add => vec + new Vector<float>(parameters[0]),
                VectorOperation.Multiply => vec * new Vector<float>(parameters[0]),
                VectorOperation.FusedMultiplyAdd => (vec * parameters[0]) + new Vector<float>(parameters[1]),
                VectorOperation.Sqrt => Vector.SquareRoot(vec),
                VectorOperation.Max => Vector.Max(vec, new Vector<float>(parameters[0])),
                VectorOperation.Min => Vector.Min(vec, new Vector<float>(parameters[0])),
                _ => vec
            };
            
            result.CopyTo(output, offset);
        });
        
        // Process remainder
        for (int i = vectorCount * vectorSize; i < input.Length; i++)
        {
            output[i] = ApplyScalarOperation(input[i], operation, parameters);
        }
    }
    
    private void ExecuteScalar(
        float[] input,
        float[] output,
        VectorOperation operation,
        float[] parameters)
    {
        Parallel.For(0, input.Length, i =>
        {
            output[i] = ApplyScalarOperation(input[i], operation, parameters);
        });
    }
    
    private float ApplyScalarOperation(
        float value,
        VectorOperation operation,
        float[] parameters)
    {
        return operation switch
        {
            VectorOperation.Add => value + parameters[0],
            VectorOperation.Multiply => value * parameters[0],
            VectorOperation.FusedMultiplyAdd => (value * parameters[0]) + parameters[1],
            VectorOperation.Sqrt => MathF.Sqrt(value),
            VectorOperation.Reciprocal => 1.0f / value,
            VectorOperation.Max => MathF.Max(value, parameters[0]),
            VectorOperation.Min => MathF.Min(value, parameters[0]),
            _ => value
        };
    }
    
    private void ProcessChunk<TIn, TOut>(
        TIn[] input,
        TOut[] output,
        Func<TIn, TOut> kernel,
        WorkChunk chunk,
        ParallelExecutionOptions options)
    {
        // Apply NUMA optimizations if available
        if (options.UseNumaOptimization && OperatingSystem.IsWindows())
        {
            SetThreadAffinity(chunk.PreferredCore);
        }
        
        for (int i = chunk.Start; i < chunk.End; i++)
        {
            output[i] = kernel(input[i]);
        }
    }
    
    private int CalculateOptimalChunkSize(int totalItems, ParallelExecutionOptions options)
    {
        // Consider cache line size (typically 64 bytes)
        const int cacheLineSize = 64;
        var itemSize = options.EstimatedItemSize ?? IntPtr.Size;
        var itemsPerCacheLine = Math.Max(1, cacheLineSize / itemSize);
        
        // Calculate based on L1 cache size (typically 32KB per core)
        const int l1CacheSize = 32 * 1024;
        var itemsPerL1Cache = l1CacheSize / itemSize;
        
        // Balance between parallelism and cache efficiency
        var minChunkSize = itemsPerCacheLine * 16; // At least 16 cache lines
        var maxChunkSize = itemsPerL1Cache / 2; // Half L1 cache
        var idealChunkSize = totalItems / (_maxParallelism * 4); // 4 chunks per core
        
        return Math.Max(minChunkSize, Math.Min(maxChunkSize, idealChunkSize));
    }
    
    private List<WorkChunk> CreateWorkChunks(int totalItems, int chunkSize)
    {
        var chunks = new List<WorkChunk>();
        var coreIndex = 0;
        
        for (int i = 0; i < totalItems; i += chunkSize)
        {
            chunks.Add(new WorkChunk
            {
                Start = i,
                End = Math.Min(i + chunkSize, totalItems),
                PreferredCore = coreIndex % _maxParallelism
            });
            coreIndex++;
        }
        
        return chunks;
    }
    
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();
    
    [DllImport("kernel32.dll")]
    private static extern UIntPtr SetThreadAffinityMask(IntPtr hThread, UIntPtr dwThreadAffinityMask);
    
    private void SetThreadAffinity(int coreIndex)
    {
        if (!OperatingSystem.IsWindows())
            return;
        
        try
        {
            var thread = GetCurrentThread();
            var mask = (UIntPtr)(1UL << coreIndex);
            SetThreadAffinityMask(thread, mask);
        }
        catch
        {
            // Ignore affinity errors
        }
    }
    
    private sealed class WorkChunk
    {
        public int Start { get; init; }
        public int End { get; init; }
        public int PreferredCore { get; init; }
    }
}

/// <summary>
/// Vector operations
/// </summary>
public enum VectorOperation
{
    Add,
    Multiply,
    FusedMultiplyAdd,
    Sqrt,
    Reciprocal,
    Max,
    Min
}

/// <summary>
/// Parallel execution options
/// </summary>
public sealed class ParallelExecutionOptions
{
    public int? MaxDegreeOfParallelism { get; set; }
    public bool UseNumaOptimization { get; set; } = true;
    public int? EstimatedItemSize { get; set; }
    public bool PreferVectorization { get; set; } = true;
}

/// <summary>
/// Extension methods for parallel operations
/// </summary>
public static class ParallelExtensions
{
    /// <summary>
    /// Parallel map with automatic batching
    /// </summary>
    public static async Task<TOut[]> ParallelMapAsync<TIn, TOut>(
        this IEnumerable<TIn> source,
        Func<TIn, Task<TOut>> mapper,
        int maxConcurrency = 0,
        CancellationToken ct = default)
        where TIn : notnull
        where TOut : notnull
    {
        var items = source.ToArray();
        var results = new TOut[items.Length];
        
        maxConcurrency = maxConcurrency > 0 ? maxConcurrency : Environment.ProcessorCount;
        using var semaphore = new SemaphoreSlim(maxConcurrency, maxConcurrency);
        
        var tasks = items.Select(async (item, index) =>
        {
            await semaphore.WaitAsync(ct);
            try
            {
                results[index] = await mapper(item);
            }
            finally
            {
                semaphore.Release();
            }
        });
        
        await Task.WhenAll(tasks);
        return results;
    }
    
    /// <summary>
    /// Parallel reduce operation
    /// </summary>
    public static async Task<T> ParallelReduceAsync<T>(
        this IEnumerable<T> source,
        Func<T, T, T> reducer,
        CancellationToken ct = default)
        where T : notnull
    {
        var items = source.ToArray();
        if (items.Length == 0)
            throw new InvalidOperationException("Cannot reduce empty sequence");
        
        if (items.Length == 1)
            return items[0];
        
        // Parallel tree reduction
        while (items.Length > 1)
        {
            var newLength = (items.Length + 1) / 2;
            var newItems = new T[newLength];
            
            await Parallel.ForAsync(0, newLength, ct, async (i, token) =>
            {
                var index1 = i * 2;
                var index2 = Math.Min(index1 + 1, items.Length - 1);
                newItems[i] = await Task.Run(() => reducer(items[index1], items[index2]), token);
            });
            
            items = newItems;
        }
        
        return items[0];
    }
}