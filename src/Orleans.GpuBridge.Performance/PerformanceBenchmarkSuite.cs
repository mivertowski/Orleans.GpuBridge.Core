using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.GCSettings;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Exporters;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Comprehensive performance benchmark suite for GPU Bridge optimizations
/// </summary>
[Config(typeof(BenchmarkConfig))]
[MemoryDiagnoser]
[ThreadingDiagnoser]
[HardwareCounters(BenchmarkDotNet.Diagnosers.HardwareCounter.CacheMisses)]
public class PerformanceBenchmarkSuite
{
    private VectorizedKernelExecutor _executor = null!;
    private HighPerformanceMemoryPool<float> _memoryPool = null!;
    private ILogger<PerformanceBenchmarkSuite> _logger = null!;
    private float[] _smallData = null!;
    private float[] _mediumData = null!;
    private float[] _largeData = null!;
    private float[] _hugeData = null!;

    [GlobalSetup]
    public void Setup()
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.SetMinimumLevel(LogLevel.Warning));
        var serviceProvider = services.BuildServiceProvider();
        
        _logger = serviceProvider.GetRequiredService<ILogger<PerformanceBenchmarkSuite>>();
        _memoryPool = new HighPerformanceMemoryPool<float>(_logger, useNumaOptimization: true);
        _executor = new VectorizedKernelExecutor(_logger, _memoryPool);
        
        // Pre-generate test data
        _smallData = GenerateData(1_000);
        _mediumData = GenerateData(100_000);
        _largeData = GenerateData(1_000_000);
        _hugeData = GenerateData(10_000_000);
        
        Console.WriteLine($"Benchmark setup completed - CPU capabilities:");
        Console.WriteLine($"  AVX-512: {_executor.HasAvx512}");
        Console.WriteLine($"  AVX2: {_executor.HasAvx2}");
        Console.WriteLine($"  FMA: {_executor.HasFma}");
        Console.WriteLine($"  NEON: {_executor.HasNeon}");
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _executor?.Dispose();
        _memoryPool?.Dispose();
    }

    #region Memory Pool Benchmarks

    [Benchmark]
    [Arguments(100)]
    [Arguments(1_000)]
    [Arguments(10_000)]
    public void MemoryPool_Rent_Return(int size)
    {
        using var memory = _memoryPool.Rent(size);
        var span = memory.Memory.Span;
        span[0] = 1.0f; // Ensure memory is used
    }

    [Benchmark]
    public void MemoryPool_Concurrent_Access()
    {
        const int iterations = 1000;
        Parallel.For(0, iterations, i =>
        {
            using var memory = _memoryPool.Rent(1000);
            var span = memory.Memory.Span;
            for (int j = 0; j < span.Length; j++)
            {
                span[j] = j;
            }
        });
    }

    [Benchmark]
    public MemoryPoolStats MemoryPool_Statistics()
    {
        return _memoryPool.GetStatistics();
    }

    #endregion

    #region Vectorized Operations Benchmarks

    [Benchmark]
    [Arguments(1_000)]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public async Task<float[]> VectorAdd_Optimized(int size)
    {
        var a = GenerateData(size);
        var b = GenerateData(size);
        return await _executor.VectorAddAsync(a, b);
    }

    [Benchmark]
    [Arguments(1_000)]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public float[] VectorAdd_Baseline(int size)
    {
        var a = GenerateData(size);
        var b = GenerateData(size);
        var result = new float[size];
        
        for (int i = 0; i < size; i++)
        {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    [Benchmark]
    [Arguments(1_000)]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public async Task<float[]> FusedMultiplyAdd_Optimized(int size)
    {
        var a = GenerateData(size);
        var b = GenerateData(size);
        var c = GenerateData(size);
        return await _executor.FusedMultiplyAddAsync(a, b, c);
    }

    [Benchmark]
    [Arguments(1_000)]
    [Arguments(100_000)]
    [Arguments(1_000_000)]
    public float[] FusedMultiplyAdd_Baseline(int size)
    {
        var a = GenerateData(size);
        var b = GenerateData(size);
        var c = GenerateData(size);
        var result = new float[size];
        
        for (int i = 0; i < size; i++)
        {
            result[i] = a[i] * b[i] + c[i];
        }
        return result;
    }

    [Benchmark]
    [Arguments(ReductionOperation.Sum)]
    [Arguments(ReductionOperation.Max)]
    [Arguments(ReductionOperation.Min)]
    public async Task<float> Reduction_Optimized_Large(ReductionOperation operation)
    {
        return await _executor.ReduceAsync(_largeData, operation);
    }

    [Benchmark]
    [Arguments(ReductionOperation.Sum)]
    [Arguments(ReductionOperation.Max)]
    [Arguments(ReductionOperation.Min)]
    public float Reduction_Baseline_Large(ReductionOperation operation)
    {
        return operation switch
        {
            ReductionOperation.Sum => _largeData.Sum(),
            ReductionOperation.Max => _largeData.Max(),
            ReductionOperation.Min => _largeData.Min(),
            _ => throw new ArgumentException($"Unknown operation: {operation}")
        };
    }

    #endregion

    #region Matrix Operations Benchmarks

    [Benchmark]
    [Arguments(64)]
    [Arguments(128)]
    [Arguments(256)]
    public async Task<float[]> MatrixMultiply_Optimized(int size)
    {
        var a = GenerateData(size * size);
        var b = GenerateData(size * size);
        return await _executor.MatrixMultiplyAsync(a, b, size, size, size);
    }

    [Benchmark]
    [Arguments(64)]
    [Arguments(128)]
    [Arguments(256)]
    public float[] MatrixMultiply_Baseline(int size)
    {
        var a = GenerateData(size * size);
        var b = GenerateData(size * size);
        var result = new float[size * size];
        
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                float sum = 0;
                for (int k = 0; k < size; k++)
                {
                    sum += a[i * size + k] * b[k * size + j];
                }
                result[i * size + j] = sum;
            }
        }
        return result;
    }

    #endregion

    #region Throughput Benchmarks

    [Benchmark]
    public async Task<long> Throughput_VectorAdd_1Second()
    {
        var stopwatch = Stopwatch.StartNew();
        long operations = 0;
        
        while (stopwatch.ElapsedMilliseconds < 1000)
        {
            await _executor.VectorAddAsync(_smallData, _smallData);
            operations++;
        }
        
        return operations;
    }

    [Benchmark]
    public async Task<long> Throughput_Reduction_1Second()
    {
        var stopwatch = Stopwatch.StartNew();
        long operations = 0;
        
        while (stopwatch.ElapsedMilliseconds < 1000)
        {
            await _executor.ReduceAsync(_mediumData, ReductionOperation.Sum);
            operations++;
        }
        
        return operations;
    }

    [Benchmark]
    public long Throughput_MemoryPool_1Second()
    {
        var stopwatch = Stopwatch.StartNew();
        long operations = 0;
        
        while (stopwatch.ElapsedMilliseconds < 1000)
        {
            using var memory = _memoryPool.Rent(1000);
            memory.Memory.Span[0] = operations;
            operations++;
        }
        
        return operations;
    }

    #endregion

    #region Scaling Benchmarks

    [Benchmark]
    [Arguments(1)]
    [Arguments(2)]
    [Arguments(4)]
    [Arguments(8)]
    public async Task<float[][]> Parallel_VectorAdd_Scaling(int parallelism)
    {
        const int batchSize = 100;
        var batches = Enumerable.Range(0, batchSize)
            .Select(_ => _mediumData)
            .ToArray();
        
        var semaphore = new SemaphoreSlim(parallelism, parallelism);
        var tasks = batches.Select(async batch =>
        {
            await semaphore.WaitAsync();
            try
            {
                return await _executor.VectorAddAsync(batch, batch);
            }
            finally
            {
                semaphore.Release();
            }
        });
        
        return await Task.WhenAll(tasks);
    }

    #endregion

    #region GC Pressure Benchmarks

    [Benchmark]
    public void GC_Pressure_NoPooling()
    {
        const int iterations = 10000;
        var arrays = new float[iterations][];
        
        for (int i = 0; i < iterations; i++)
        {
            arrays[i] = new float[1000];
            arrays[i][0] = i;
        }
        
        // Force GC to measure pressure
        var gen0Before = GC.CollectionCount(0);
        var gen1Before = GC.CollectionCount(1);
        
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        
        var gen0After = GC.CollectionCount(0);
        var gen1After = GC.CollectionCount(1);
        
        Console.WriteLine($"GC Collections - Gen0: {gen0After - gen0Before}, Gen1: {gen1After - gen1Before}");
    }

    [Benchmark]
    public void GC_Pressure_WithPooling()
    {
        const int iterations = 10000;
        var memories = new IMemoryOwner<float>[iterations];
        
        for (int i = 0; i < iterations; i++)
        {
            memories[i] = _memoryPool.Rent(1000);
            memories[i].Memory.Span[0] = i;
        }
        
        // Dispose all at once
        for (int i = 0; i < iterations; i++)
        {
            memories[i].Dispose();
        }
        
        var gen0Before = GC.CollectionCount(0);
        var gen1Before = GC.CollectionCount(1);
        
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        
        var gen0After = GC.CollectionCount(0);
        var gen1After = GC.CollectionCount(1);
        
        Console.WriteLine($"GC Collections - Gen0: {gen0After - gen0Before}, Gen1: {gen1After - gen1Before}");
    }

    #endregion

    #region Helper Methods

    private float[] GenerateData(int size)
    {
        var data = new float[size];
        var random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)random.NextDouble() * 100;
        }
        
        return data;
    }

    #endregion

    #region Benchmark Configuration

    public class BenchmarkConfig : ManualConfig
    {
        public BenchmarkConfig()
        {
            AddJob(Job.Default
                .WithGcServer(true)
                .WithGcConcurrent(true)
                .WithGcForce(false));
            
            AddColumn(
                StatisticColumn.Mean,
                StatisticColumn.Min,
                StatisticColumn.Max,
                StatisticColumn.P95,
                BaselineColumn.RatioMean,
                RankColumn.Arabic);
            
            AddExporter(MarkdownExporter.GitHub);
            AddExporter(HtmlExporter.Default);
        }
    }

    #endregion
}

/// <summary>
/// Performance analysis and reporting utilities
/// </summary>
public static class PerformanceAnalyzer
{
    public static async Task<PerformanceReport> GenerateReportAsync(
        VectorizedKernelExecutor executor,
        HighPerformanceMemoryPool<float> memoryPool)
    {
        var report = new PerformanceReport
        {
            GeneratedAt = DateTime.UtcNow,
            SystemInfo = GetSystemInfo(),
            CpuCapabilities = GetCpuCapabilities(executor),
            MemoryPoolStats = memoryPool.GetStatistics()
        };

        // Run performance tests
        var tests = new List<Task<PerformanceTestResult>>
        {
            RunVectorAddTest(executor),
            RunReductionTest(executor),
            RunMemoryPoolTest(memoryPool),
            RunThroughputTest(executor)
        };

        report.TestResults = await Task.WhenAll(tests);
        
        // Calculate overall performance score
        report.OverallScore = CalculatePerformanceScore(report.TestResults);
        
        return report;
    }

    private static async Task<PerformanceTestResult> RunVectorAddTest(VectorizedKernelExecutor executor)
    {
        const int iterations = 100;
        const int dataSize = 1_000_000;
        
        var data1 = GenerateTestData(dataSize);
        var data2 = GenerateTestData(dataSize);
        var times = new List<double>();
        
        // Warmup
        await executor.VectorAddAsync(data1, data2);
        
        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            await executor.VectorAddAsync(data1, data2);
            sw.Stop();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        
        return new PerformanceTestResult
        {
            TestName = "VectorAdd",
            Iterations = iterations,
            DataSize = dataSize,
            AverageTimeMs = times.Average(),
            MinTimeMs = times.Min(),
            MaxTimeMs = times.Max(),
            ThroughputElementsPerSec = dataSize / (times.Average() / 1000.0)
        };
    }

    private static async Task<PerformanceTestResult> RunReductionTest(VectorizedKernelExecutor executor)
    {
        const int iterations = 1000;
        const int dataSize = 100_000;
        
        var data = GenerateTestData(dataSize);
        var times = new List<double>();
        
        // Warmup
        await executor.ReduceAsync(data, ReductionOperation.Sum);
        
        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            await executor.ReduceAsync(data, ReductionOperation.Sum);
            sw.Stop();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        
        return new PerformanceTestResult
        {
            TestName = "Reduction",
            Iterations = iterations,
            DataSize = dataSize,
            AverageTimeMs = times.Average(),
            MinTimeMs = times.Min(),
            MaxTimeMs = times.Max(),
            ThroughputElementsPerSec = dataSize / (times.Average() / 1000.0)
        };
    }

    private static Task<PerformanceTestResult> RunMemoryPoolTest(HighPerformanceMemoryPool<float> pool)
    {
        const int iterations = 100_000;
        const int bufferSize = 1000;
        
        var times = new List<double>();
        
        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            using var memory = pool.Rent(bufferSize);
            memory.Memory.Span[0] = i;
            sw.Stop();
            times.Add(sw.Elapsed.TotalMicroseconds);
        }
        
        return Task.FromResult(new PerformanceTestResult
        {
            TestName = "MemoryPool",
            Iterations = iterations,
            DataSize = bufferSize,
            AverageTimeMs = times.Average() / 1000.0,
            MinTimeMs = times.Min() / 1000.0,
            MaxTimeMs = times.Max() / 1000.0,
            ThroughputElementsPerSec = 1_000_000 / times.Average() // Operations per second
        });
    }

    private static async Task<PerformanceTestResult> RunThroughputTest(VectorizedKernelExecutor executor)
    {
        const int testDurationMs = 5000;
        const int dataSize = 10_000;
        
        var data1 = GenerateTestData(dataSize);
        var data2 = GenerateTestData(dataSize);
        var operations = 0L;
        
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < testDurationMs)
        {
            await executor.VectorAddAsync(data1, data2);
            operations++;
        }
        sw.Stop();
        
        return new PerformanceTestResult
        {
            TestName = "Throughput",
            Iterations = (int)operations,
            DataSize = dataSize,
            AverageTimeMs = sw.ElapsedMilliseconds / (double)operations,
            ThroughputElementsPerSec = operations * dataSize / (sw.ElapsedMilliseconds / 1000.0)
        };
    }

    private static float[] GenerateTestData(int size)
    {
        var data = new float[size];
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return data;
    }

    private static SystemInfo GetSystemInfo()
    {
        return new SystemInfo
        {
            ProcessorCount = Environment.ProcessorCount,
            WorkingSet = Environment.WorkingSet,
            GcTotalMemory = GC.GetTotalMemory(false),
            GcLatencyMode = GCSettings.LatencyMode.ToString(),
            GcIsServerGc = GCSettings.IsServerGC
        };
    }

    private static CpuCapabilities GetCpuCapabilities(VectorizedKernelExecutor executor)
    {
        return new CpuCapabilities
        {
            HasAvx512 = executor.HasAvx512,
            HasAvx2 = executor.HasAvx2,
            HasAvx = executor.HasAvx,
            HasFma = executor.HasFma,
            HasNeon = executor.HasNeon
        };
    }

    private static double CalculatePerformanceScore(PerformanceTestResult[] results)
    {
        // Simple scoring algorithm - can be enhanced
        var scores = results.Select(r => Math.Min(r.ThroughputElementsPerSec / 1_000_000, 100)).ToArray();
        return scores.Average();
    }
}

#region Performance Report Classes

public record PerformanceReport
{
    public DateTime GeneratedAt { get; init; }
    public SystemInfo SystemInfo { get; init; } = null!;
    public CpuCapabilities CpuCapabilities { get; init; } = null!;
    public MemoryPoolStats MemoryPoolStats { get; init; } = null!;
    public PerformanceTestResult[] TestResults { get; init; } = Array.Empty<PerformanceTestResult>();
    public double OverallScore { get; init; }
}

public record SystemInfo
{
    public int ProcessorCount { get; init; }
    public long WorkingSet { get; init; }
    public long GcTotalMemory { get; init; }
    public string GcLatencyMode { get; init; } = string.Empty;
    public bool GcIsServerGc { get; init; }
}

public record CpuCapabilities
{
    public bool HasAvx512 { get; init; }
    public bool HasAvx2 { get; init; }
    public bool HasAvx { get; init; }
    public bool HasFma { get; init; }
    public bool HasNeon { get; init; }
}

public record PerformanceTestResult
{
    public string TestName { get; init; } = string.Empty;
    public int Iterations { get; init; }
    public int DataSize { get; init; }
    public double AverageTimeMs { get; init; }
    public double MinTimeMs { get; init; }
    public double MaxTimeMs { get; init; }
    public double ThroughputElementsPerSec { get; init; }
}

#endregion