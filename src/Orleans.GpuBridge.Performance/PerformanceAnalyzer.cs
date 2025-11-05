using System.Diagnostics;
using Orleans.GpuBridge.Performance.Models;

namespace Orleans.GpuBridge.Performance;

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
            GcLatencyMode = System.Runtime.GCSettings.LatencyMode.ToString(),
            GcIsServerGc = System.Runtime.GCSettings.IsServerGC
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
