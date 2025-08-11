using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.DotCompute;
using Orleans.GpuBridge.Runtime;
using Orleans.GpuBridge.Runtime.BackendProviders;
using Xunit;
using Xunit.Abstractions;

namespace Orleans.GpuBridge.Tests.Benchmarks;

public class PerformanceBenchmarks
{
    private readonly ITestOutputHelper _output;
    
    public PerformanceBenchmarks(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public async Task Benchmark_Memory_Pool_Allocation()
    {
        // Benchmark memory pool allocation performance
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        var poolManager = new MemoryPoolManager(loggerFactory);
        var pool = poolManager.GetPool<float>();
        
        const int iterations = 10000;
        var sizes = new[] { 100, 1000, 10000, 100000 };
        
        foreach (var size in sizes)
        {
            var sw = Stopwatch.StartNew();
            
            for (int i = 0; i < iterations; i++)
            {
                using var memory = pool.Rent(size);
                // Simulate some work
                var span = memory.AsMemory().Span;
                span[0] = 1.0f;
            }
            
            sw.Stop();
            var throughput = iterations / sw.Elapsed.TotalSeconds;
            _output.WriteLine($"Memory Pool Allocation (size={size}): {throughput:N0} ops/sec");
            
            // Assert reasonable performance (at least 10K ops/sec)
            Assert.True(throughput > 10000, $"Memory pool too slow: {throughput:N0} ops/sec");
        }
    }

    [Fact]
    public async Task Benchmark_Vectorized_Operations()
    {
        // Benchmark SIMD vectorized operations
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
        var executor = new ParallelKernelExecutor(logger);
        
        var sizes = new[] { 1000, 10000, 100000, 1000000 };
        var operations = new[]
        {
            (VectorOperation.Add, "Add"),
            (VectorOperation.Multiply, "Multiply"),
            (VectorOperation.FusedMultiplyAdd, "FMA"),
            (VectorOperation.Sqrt, "Sqrt")
        };
        
        foreach (var size in sizes)
        {
            var input = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
            
            foreach (var (operation, name) in operations)
            {
                var parameters = operation == VectorOperation.FusedMultiplyAdd 
                    ? new[] { 2.0f, 3.0f } 
                    : new[] { 2.0f };
                
                var sw = Stopwatch.StartNew();
                var result = await executor.ExecuteVectorizedAsync(input, operation, parameters);
                sw.Stop();
                
                var throughput = size / sw.Elapsed.TotalSeconds;
                var bandwidth = (size * sizeof(float) * 2) / (sw.Elapsed.TotalSeconds * 1024 * 1024); // MB/s
                
                _output.WriteLine($"Vectorized {name} (size={size}): {throughput:N0} ops/sec, {bandwidth:N0} MB/s");
                
                // Assert minimum performance
                Assert.True(throughput > size, $"Vectorized operation too slow: {throughput:N0} ops/sec");
            }
        }
    }

    [Fact]
    public async Task Benchmark_Buffer_Serialization()
    {
        // Benchmark serialization performance
        var sizes = new[] { 100, 1000, 10000, 100000 };
        
        foreach (var size in sizes)
        {
            var data = Enumerable.Range(1, size).Select(i => (float)i).ToArray();
            
            // Benchmark serialization
            var swSerialize = Stopwatch.StartNew();
            var serialized = BufferSerializer.Serialize<float>(data);
            swSerialize.Stop();
            
            // Benchmark deserialization
            var swDeserialize = Stopwatch.StartNew();
            var deserialized = BufferSerializer.Deserialize<float>(serialized);
            swDeserialize.Stop();
            
            var serializeThroughput = (size * sizeof(float)) / (swSerialize.Elapsed.TotalSeconds * 1024 * 1024);
            var deserializeThroughput = (size * sizeof(float)) / (swDeserialize.Elapsed.TotalSeconds * 1024 * 1024);
            
            _output.WriteLine($"Serialization (size={size}): {serializeThroughput:N0} MB/s");
            _output.WriteLine($"Deserialization (size={size}): {deserializeThroughput:N0} MB/s");
            
            // Assert minimum throughput (at least 100 MB/s)
            Assert.True(serializeThroughput > 100, $"Serialization too slow: {serializeThroughput:N0} MB/s");
            Assert.True(deserializeThroughput > 100, $"Deserialization too slow: {deserializeThroughput:N0} MB/s");
        }
    }

    [Fact]
    public async Task Benchmark_Compressed_Serialization()
    {
        // Benchmark compressed serialization
        var data = new float[100000];
        Array.Fill(data, 1.0f); // Highly compressible
        
        var swCompress = Stopwatch.StartNew();
        var compressed = await BufferSerializer.SerializeCompressedAsync(data, Orleans.GpuBridge.DotCompute.CompressionLevel.Optimal);
        swCompress.Stop();
        
        var swDecompress = Stopwatch.StartNew();
        var decompressed = await BufferSerializer.DeserializeCompressedAsync<float>(compressed);
        swDecompress.Stop();
        
        var compressionRatio = (data.Length * sizeof(float)) / (double)compressed.Length;
        var compressThroughput = (data.Length * sizeof(float)) / (swCompress.Elapsed.TotalSeconds * 1024 * 1024);
        var decompressThroughput = (data.Length * sizeof(float)) / (swDecompress.Elapsed.TotalSeconds * 1024 * 1024);
        
        _output.WriteLine($"Compression ratio: {compressionRatio:N2}x");
        _output.WriteLine($"Compress throughput: {compressThroughput:N0} MB/s");
        _output.WriteLine($"Decompress throughput: {decompressThroughput:N0} MB/s");
        
        // Assert reasonable compression ratio and throughput
        Assert.True(compressionRatio > 2, $"Poor compression ratio: {compressionRatio:N2}x");
        Assert.True(compressThroughput > 10, $"Compression too slow: {compressThroughput:N0} MB/s");
    }

    [Fact]
    public async Task Benchmark_Parallel_Execution()
    {
        // Benchmark parallel execution scaling
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
        var executor = new ParallelKernelExecutor(logger);
        
        var input = Enumerable.Range(1, 1000000).ToArray();
        Func<int, int> kernel = x => 
        {
            // Simulate some computation
            var result = x;
            for (int i = 0; i < 10; i++)
                result = (result * 7 + 13) % 1000000;
            return result;
        };
        
        var parallelismLevels = new[] { 1, 2, 4, 8, Environment.ProcessorCount };
        
        foreach (var parallelism in parallelismLevels)
        {
            var options = new ParallelExecutionOptions
            {
                MaxDegreeOfParallelism = parallelism
            };
            
            var sw = Stopwatch.StartNew();
            var result = await executor.ExecuteAsync(input, kernel, options);
            sw.Stop();
            
            var throughput = input.Length / sw.Elapsed.TotalSeconds;
            _output.WriteLine($"Parallel execution (parallelism={parallelism}): {throughput:N0} ops/sec");
            
            // Verify result correctness (spot check)
            Assert.Equal(kernel(input[0]), result[0]);
            Assert.Equal(kernel(input[input.Length - 1]), result[result.Length - 1]);
        }
    }

    [Fact]
    public async Task Benchmark_Backend_Provider_Initialization()
    {
        // Benchmark backend provider initialization time
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        var logger = serviceProvider.GetRequiredService<ILogger<BackendProviderFactory>>();
        
        var sw = Stopwatch.StartNew();
        var factory = new BackendProviderFactory(serviceProvider, logger);
        factory.Initialize();
        sw.Stop();
        
        _output.WriteLine($"Backend provider initialization: {sw.ElapsedMilliseconds}ms");
        
        // Assert reasonable initialization time (< 1 second)
        Assert.True(sw.ElapsedMilliseconds < 1000, $"Initialization too slow: {sw.ElapsedMilliseconds}ms");
        
        // Benchmark context creation
        var provider = factory.GetPrimaryProvider();
        
        sw.Restart();
        const int iterations = 1000;
        for (int i = 0; i < iterations; i++)
        {
            using var context = provider.CreateContext();
        }
        sw.Stop();
        
        var contextCreationRate = iterations / sw.Elapsed.TotalSeconds;
        _output.WriteLine($"Context creation rate: {contextCreationRate:N0} contexts/sec");
        
        // Assert reasonable context creation rate
        Assert.True(contextCreationRate > 1000, $"Context creation too slow: {contextCreationRate:N0}/sec");
    }

    [Fact]
    public async Task Benchmark_Device_Detection()
    {
        // Benchmark device detection performance
        var services = new ServiceCollection();
        services.AddLogging();
        services.Configure<GpuBridgeOptions>(opt =>
        {
            opt.PreferGpu = true;
            opt.MemoryPoolSizeMB = 1024;
        });
        
        var serviceProvider = services.BuildServiceProvider();
        var logger = serviceProvider.GetRequiredService<ILogger<DeviceBroker>>();
        var options = serviceProvider.GetRequiredService<Microsoft.Extensions.Options.IOptions<GpuBridgeOptions>>();
        
        var sw = Stopwatch.StartNew();
        using var broker = new DeviceBroker(logger, options);
        await broker.InitializeAsync(default);
        sw.Stop();
        
        _output.WriteLine($"Device detection time: {sw.ElapsedMilliseconds}ms");
        _output.WriteLine($"Devices found: {broker.DeviceCount}");
        
        // Assert reasonable detection time (< 2 seconds)
        Assert.True(sw.ElapsedMilliseconds < 2000, $"Device detection too slow: {sw.ElapsedMilliseconds}ms");
        
        // Benchmark device selection
        sw.Restart();
        const int iterations = 10000;
        for (int i = 0; i < iterations; i++)
        {
            var device = broker.GetBestDevice();
        }
        sw.Stop();
        
        var selectionRate = iterations / sw.Elapsed.TotalSeconds;
        _output.WriteLine($"Device selection rate: {selectionRate:N0} selections/sec");
        
        // Assert high selection rate (should be very fast)
        Assert.True(selectionRate > 100000, $"Device selection too slow: {selectionRate:N0}/sec");
    }

    [Fact]
    public void Benchmark_Memory_Copy_Performance()
    {
        // Benchmark memory copy performance for different sizes
        var sizes = new[] { 1024, 16384, 262144, 4194304 }; // 1KB, 16KB, 256KB, 4MB
        
        foreach (var size in sizes)
        {
            var source = new byte[size];
            var destination = new byte[size];
            Random.Shared.NextBytes(source);
            
            const int iterations = 1000;
            var sw = Stopwatch.StartNew();
            
            for (int i = 0; i < iterations; i++)
            {
                Buffer.BlockCopy(source, 0, destination, 0, size);
            }
            
            sw.Stop();
            var bandwidth = (size * iterations) / (sw.Elapsed.TotalSeconds * 1024 * 1024 * 1024); // GB/s
            _output.WriteLine($"Memory copy bandwidth (size={size/1024}KB): {bandwidth:N2} GB/s");
            
            // Assert reasonable memory bandwidth (> 1 GB/s)
            Assert.True(bandwidth > 1, $"Memory bandwidth too low: {bandwidth:N2} GB/s");
        }
    }

    [Fact]
    public async Task Benchmark_End_To_End_Pipeline()
    {
        // Benchmark complete pipeline from input to output
        var services = new ServiceCollection();
        services.AddLogging();
        var serviceProvider = services.BuildServiceProvider();
        
        // Setup components
        var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
        var poolManager = new MemoryPoolManager(loggerFactory);
        var pool = poolManager.GetPool<float>();
        
        var execLogger = serviceProvider.GetRequiredService<ILogger<ParallelKernelExecutor>>();
        var executor = new ParallelKernelExecutor(execLogger);
        
        var data = Enumerable.Range(1, 100000).Select(i => (float)i).ToArray();
        
        var sw = Stopwatch.StartNew();
        
        // Complete pipeline
        using (var memory = pool.Rent(data.Length))
        {
            // Write data to memory
            memory.AsMemory().Span[..data.Length].CopyTo(data);
            
            // Simulate GPU transfer
            await memory.CopyToDeviceAsync();
            
            // Execute computation
            var result = await executor.ExecuteVectorizedAsync(
                data, VectorOperation.FusedMultiplyAdd, new[] { 2.0f, 3.0f });
            
            // Serialize result
            var serialized = BufferSerializer.Serialize<float>(result);
            
            // Simulate network transfer with compression
            var compressed = await BufferSerializer.SerializeCompressedAsync(result);
            
            // Deserialize
            var final = await BufferSerializer.DeserializeCompressedAsync<float>(compressed);
            
            // Simulate GPU transfer back
            await memory.CopyFromDeviceAsync();
        }
        
        sw.Stop();
        
        var throughput = data.Length / sw.Elapsed.TotalSeconds;
        _output.WriteLine($"End-to-end pipeline throughput: {throughput:N0} elements/sec");
        _output.WriteLine($"End-to-end pipeline time: {sw.ElapsedMilliseconds}ms");
        
        // Assert reasonable end-to-end performance
        Assert.True(throughput > 100000, $"Pipeline too slow: {throughput:N0} elements/sec");
    }
}