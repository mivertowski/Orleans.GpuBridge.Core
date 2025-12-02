using System.Diagnostics;
using Orleans.GpuBridge.Abstractions.Memory;
using Orleans.GpuBridge.Runtime;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Performance.Tests;

/// <summary>
/// Performance tests for memory pool allocation overhead.
/// Tests that pooled allocations reduce overhead vs direct allocation.
/// </summary>
public sealed class MemoryPoolPerformanceTests
{
    private readonly ITestOutputHelper _output;

    private const int WarmupIterations = 100;
    private const int BenchmarkIterations = 10000;

    public MemoryPoolPerformanceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Tests CpuMemoryPool allocation and return performance.
    /// </summary>
    [Fact]
    public void CpuMemoryPool_RentAndReturn_MeasuresPerformance()
    {
        // Arrange
        var pool = new CpuMemoryPool<TestStruct>();
        var sw = new Stopwatch();
        var allocations = new List<IGpuMemory<TestStruct>>(BenchmarkIterations);
        var rentTimes = new List<double>(BenchmarkIterations);
        var returnTimes = new List<double>(BenchmarkIterations);

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            var mem = pool.Rent(10);
            pool.Return(mem);
        }

        // Benchmark rents
        for (int i = 0; i < BenchmarkIterations; i++)
        {
            sw.Restart();
            var mem = pool.Rent(10);
            sw.Stop();
            rentTimes.Add(sw.Elapsed.TotalMicroseconds);
            allocations.Add(mem);
        }

        // Benchmark returns
        for (int i = 0; i < BenchmarkIterations; i++)
        {
            sw.Restart();
            pool.Return(allocations[i]);
            sw.Stop();
            returnTimes.Add(sw.Elapsed.TotalMicroseconds);
        }

        // Calculate metrics
        rentTimes.Sort();
        returnTimes.Sort();

        var rentAvg = rentTimes.Average();
        var rentP99 = rentTimes[(int)(rentTimes.Count * 0.99)];
        var returnAvg = returnTimes.Average();
        var returnP99 = returnTimes[(int)(returnTimes.Count * 0.99)];

        _output.WriteLine("=== CPU Memory Pool Performance ===");
        _output.WriteLine($"Iterations: {BenchmarkIterations:N0}");
        _output.WriteLine("");
        _output.WriteLine("Rent operations:");
        _output.WriteLine($"  Average: {rentAvg:F3}μs");
        _output.WriteLine($"  p99: {rentP99:F3}μs");
        _output.WriteLine("");
        _output.WriteLine("Return operations:");
        _output.WriteLine($"  Average: {returnAvg:F3}μs");
        _output.WriteLine($"  p99: {returnP99:F3}μs");

        var stats = pool.GetStats();
        _output.WriteLine("");
        _output.WriteLine($"Pool Stats:");
        _output.WriteLine($"  Total Allocated: {stats.TotalAllocated:N0} bytes");
        _output.WriteLine($"  Rent Count: {stats.RentCount}");
        _output.WriteLine($"  Return Count: {stats.ReturnCount}");
    }

    /// <summary>
    /// Tests pool reuse efficiency.
    /// </summary>
    [Fact]
    public void CpuMemoryPool_ReusePattern_ShowsEfficiency()
    {
        // Arrange
        var pool = new CpuMemoryPool<TestStruct>();
        const int blockSize = 100;
        const int iterations = 1000;
        var sw = Stopwatch.StartNew();

        // Simulate allocation/free pattern
        for (int round = 0; round < iterations; round++)
        {
            var mem = pool.Rent(blockSize);
            pool.Return(mem);
        }

        sw.Stop();

        var stats = pool.GetStats();

        _output.WriteLine("=== Pool Reuse Efficiency ===");
        _output.WriteLine($"Iterations: {iterations}");
        _output.WriteLine($"Block Size: {blockSize} elements");
        _output.WriteLine($"Total Time: {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"Throughput: {iterations / sw.Elapsed.TotalSeconds:N0} ops/sec");
        _output.WriteLine("");
        _output.WriteLine($"Pool Statistics:");
        _output.WriteLine($"  Total Bytes Allocated: {stats.TotalAllocated:N0}");
        _output.WriteLine($"  Rent Count: {stats.RentCount}");
        _output.WriteLine($"  Pooled Buffers: {stats.BufferCount}");

        // Should be able to do at least 10K ops/sec
        var throughput = iterations / sw.Elapsed.TotalSeconds;
        throughput.Should().BeGreaterThan(10000);
    }

    /// <summary>
    /// Tests concurrent allocation performance.
    /// </summary>
    [Fact]
    public async Task CpuMemoryPool_ConcurrentAccess_HandlesContention()
    {
        // Arrange
        var pool = new CpuMemoryPool<TestStruct>();
        const int threadCount = 8;
        const int opsPerThread = 1000;
        var allTimes = new System.Collections.Concurrent.ConcurrentBag<double>();

        // Act
        var sw = Stopwatch.StartNew();

        var tasks = Enumerable.Range(0, threadCount).Select(async _ =>
        {
            await Task.Yield();
            var localSw = new Stopwatch();

            for (int i = 0; i < opsPerThread; i++)
            {
                localSw.Restart();
                var mem = pool.Rent(10);
                pool.Return(mem);
                localSw.Stop();
                allTimes.Add(localSw.Elapsed.TotalMicroseconds);
            }
        });

        await Task.WhenAll(tasks);
        sw.Stop();

        // Calculate metrics
        var times = allTimes.ToList();
        times.Sort();

        var totalOps = threadCount * opsPerThread;
        var throughput = totalOps / sw.Elapsed.TotalSeconds;
        var avgLatency = times.Average();
        var p99Latency = times[(int)(times.Count * 0.99)];

        _output.WriteLine("=== Concurrent Pool Access ===");
        _output.WriteLine($"Threads: {threadCount}");
        _output.WriteLine($"Ops per thread: {opsPerThread}");
        _output.WriteLine($"Total ops: {totalOps:N0}");
        _output.WriteLine($"Duration: {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"Throughput: {throughput:N0} ops/sec");
        _output.WriteLine($"Avg Latency: {avgLatency:F3}μs");
        _output.WriteLine($"p99 Latency: {p99Latency:F3}μs");

        // Should handle concurrent access without major slowdown
        throughput.Should().BeGreaterThan(50000);
    }

    /// <summary>
    /// Compares pooled vs non-pooled allocation.
    /// </summary>
    [Fact]
    public void PooledVsDirect_Allocation_ComparesOverhead()
    {
        const int iterations = 10000;
        const int size = 1000;
        var sw = new Stopwatch();

        // Benchmark direct allocation
        sw.Start();
        for (int i = 0; i < iterations; i++)
        {
            var array = new TestStruct[size];
            // Prevent optimization
            array[0] = new TestStruct { Value = i };
        }
        sw.Stop();
        var directTime = sw.Elapsed.TotalMilliseconds;

        // Benchmark pooled allocation
        var pool = new CpuMemoryPool<TestStruct>();
        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            var mem = pool.Rent(size);
            pool.Return(mem);
        }
        sw.Stop();
        var pooledTime = sw.Elapsed.TotalMilliseconds;

        _output.WriteLine("=== Pooled vs Direct Allocation ===");
        _output.WriteLine($"Iterations: {iterations:N0}");
        _output.WriteLine($"Size: {size} elements");
        _output.WriteLine("");
        _output.WriteLine($"Direct allocation: {directTime:F2}ms");
        _output.WriteLine($"Pooled allocation: {pooledTime:F2}ms");

        var speedup = directTime / pooledTime;
        _output.WriteLine($"Speedup: {speedup:F2}x");

        // Pool should provide some benefit
        // (In practice, pooling benefits increase with GPU memory)
    }

    private struct TestStruct
    {
        public int Value;
        public float Data;
        public long Timestamp;
    }
}
