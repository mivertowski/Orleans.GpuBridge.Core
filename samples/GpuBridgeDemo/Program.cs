using System.CommandLine;
using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.BridgeFX;
using Orleans.GpuBridge.Runtime;
using Orleans.Hosting;

var rootCommand = new RootCommand("Orleans GPU Bridge Demo - Demonstrates GPU acceleration capabilities");

// Vector addition demo
var vectorAddCommand = new Command("vector-add", "Perform GPU-accelerated vector addition")
{
    new Option<int>("--size", () => 1000000, "Size of vectors"),
    new Option<bool>("--cpu", () => false, "Force CPU execution for comparison")
};

vectorAddCommand.SetHandler(async (int size, bool forceCpu) =>
{
    Console.WriteLine($"\n🚀 Vector Addition Demo - Size: {size:N0} elements\n");
    
    using var host = CreateHost(forceCpu);
    await host.StartAsync();
    
    var grainFactory = host.Services.GetRequiredService<IGrainFactory>();
    var grain = grainFactory.GetGrain<IVectorComputeGrain>(0);
    
    // Generate test data
    var a = GenerateRandomArray(size);
    var b = GenerateRandomArray(size);
    
    // Warm-up
    Console.WriteLine("Warming up...");
    await grain.AddVectorsAsync(a.Take(1000).ToArray(), b.Take(1000).ToArray());
    
    // Benchmark
    Console.WriteLine($"Processing {size:N0} elements on {(forceCpu ? "CPU" : "GPU")}...\n");
    
    var sw = Stopwatch.StartNew();
    var result = await grain.AddVectorsAsync(a, b);
    sw.Stop();
    
    // Verify results
    var isCorrect = VerifyVectorAddition(a, b, result);
    
    Console.WriteLine($"✅ Execution Time: {sw.ElapsedMilliseconds:N0} ms");
    Console.WriteLine($"✅ Throughput: {(size * 2.0 / sw.Elapsed.TotalSeconds / 1_000_000):F2} million ops/sec");
    Console.WriteLine($"✅ Bandwidth: {(size * 3 * sizeof(float) / sw.Elapsed.TotalSeconds / 1_073_741_824):F2} GB/s");
    Console.WriteLine($"✅ Verification: {(isCorrect ? "PASSED" : "FAILED")}");
    
    await host.StopAsync();
}, new InvocationContext().BindingContext.ParseResult.GetValueForOption<int>("--size"),
   new InvocationContext().BindingContext.ParseResult.GetValueForOption<bool>("--cpu"));

// Matrix multiplication demo
var matrixMultCommand = new Command("matrix-mult", "Perform GPU-accelerated matrix multiplication")
{
    new Option<int>("--size", () => 512, "Size of square matrices"),
    new Option<bool>("--cpu", () => false, "Force CPU execution for comparison")
};

matrixMultCommand.SetHandler(async (int size, bool forceCpu) =>
{
    Console.WriteLine($"\n🚀 Matrix Multiplication Demo - Size: {size}x{size}\n");
    
    using var host = CreateHost(forceCpu);
    await host.StartAsync();
    
    var grainFactory = host.Services.GetRequiredService<IGrainFactory>();
    var grain = grainFactory.GetGrain<IMatrixComputeGrain>(0);
    
    // Generate test matrices
    var a = GenerateRandomMatrix(size);
    var b = GenerateRandomMatrix(size);
    
    Console.WriteLine($"Processing {size}x{size} matrices on {(forceCpu ? "CPU" : "GPU")}...\n");
    
    var sw = Stopwatch.StartNew();
    var result = await grain.MultiplyMatricesAsync(a, b);
    sw.Stop();
    
    var gflops = (2.0 * size * size * size) / (sw.Elapsed.TotalSeconds * 1_000_000_000);
    
    Console.WriteLine($"✅ Execution Time: {sw.ElapsedMilliseconds:N0} ms");
    Console.WriteLine($"✅ Performance: {gflops:F2} GFLOPS");
    Console.WriteLine($"✅ Matrix Size: {size}x{size} ({size * size:N0} elements)");
    
    await host.StopAsync();
}, new InvocationContext().BindingContext.ParseResult.GetValueForOption<int>("--size"),
   new InvocationContext().BindingContext.ParseResult.GetValueForOption<bool>("--cpu"));

// Image processing demo
var imageProcessCommand = new Command("image", "Perform GPU-accelerated image processing")
{
    new Option<int>("--width", () => 1920, "Image width"),
    new Option<int>("--height", () => 1080, "Image height"),
    new Option<string>("--filter", () => "blur", "Filter type: blur, edge, sharpen")
};

imageProcessCommand.SetHandler(async (int width, int height, string filter) =>
{
    Console.WriteLine($"\n🖼️ Image Processing Demo - {width}x{height} - Filter: {filter}\n");
    
    using var host = CreateHost(false);
    await host.StartAsync();
    
    var grainFactory = host.Services.GetRequiredService<IGrainFactory>();
    var grain = grainFactory.GetGrain<IImageProcessingGrain>(0);
    
    // Generate test image
    var pixels = GenerateRandomImage(width, height);
    
    Console.WriteLine($"Processing {width}x{height} image ({width * height:N0} pixels)...\n");
    
    var sw = Stopwatch.StartNew();
    var result = filter.ToLower() switch
    {
        "blur" => await grain.ApplyGaussianBlurAsync(pixels, width, height),
        "edge" => await grain.DetectEdgesAsync(pixels, width, height),
        "sharpen" => await grain.SharpenImageAsync(pixels, width, height),
        _ => pixels
    };
    sw.Stop();
    
    var megapixelsPerSecond = (width * height) / (sw.Elapsed.TotalSeconds * 1_000_000);
    
    Console.WriteLine($"✅ Execution Time: {sw.ElapsedMilliseconds:N0} ms");
    Console.WriteLine($"✅ Throughput: {megapixelsPerSecond:F2} megapixels/sec");
    Console.WriteLine($"✅ Pixels Processed: {width * height:N0}");
    
    await host.StopAsync();
}, new InvocationContext().BindingContext.ParseResult.GetValueForOption<int>("--width"),
   new InvocationContext().BindingContext.ParseResult.GetValueForOption<int>("--height"),
   new InvocationContext().BindingContext.ParseResult.GetValueForOption<string>("--filter"));

// Benchmark command
var benchmarkCommand = new Command("benchmark", "Run comprehensive GPU benchmarks");

benchmarkCommand.SetHandler(async () =>
{
    Console.WriteLine("\n📊 Comprehensive GPU Benchmark Suite\n");
    Console.WriteLine("=====================================\n");
    
    using var host = CreateHost(false);
    await host.StartAsync();
    
    var grainFactory = host.Services.GetRequiredService<IGrainFactory>();
    
    // Vector operations benchmark
    Console.WriteLine("1. Vector Operations:");
    await BenchmarkVectorOps(grainFactory);
    
    // Matrix operations benchmark
    Console.WriteLine("\n2. Matrix Operations:");
    await BenchmarkMatrixOps(grainFactory);
    
    // Reduction operations benchmark
    Console.WriteLine("\n3. Reduction Operations:");
    await BenchmarkReductions(grainFactory);
    
    await host.StopAsync();
});

rootCommand.AddCommand(vectorAddCommand);
rootCommand.AddCommand(matrixMultCommand);
rootCommand.AddCommand(imageProcessCommand);
rootCommand.AddCommand(benchmarkCommand);

// Execute command
return await rootCommand.InvokeAsync(args);

// Helper methods
static IHost CreateHost(bool forceCpu)
{
    return new HostBuilder()
        .UseOrleans(builder =>
        {
            builder
                .UseLocalhostClustering()
                .ConfigureServices(services =>
                {
                    services.AddGpuBridge(options =>
                    {
                        options.PreferGpu = !forceCpu;
                        options.MemoryPoolSizeMB = 1024;
                        options.EnableHealthChecks = true;
                        options.EnableTelemetry = true;
                    });
                });
        })
        .ConfigureLogging(builder =>
        {
            builder.SetMinimumLevel(LogLevel.Warning);
            builder.AddConsole();
        })
        .Build();
}

static float[] GenerateRandomArray(int size)
{
    var random = new Random();
    var array = new float[size];
    for (int i = 0; i < size; i++)
    {
        array[i] = (float)random.NextDouble() * 100;
    }
    return array;
}

static float[,] GenerateRandomMatrix(int size)
{
    var random = new Random();
    var matrix = new float[size, size];
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i, j] = (float)random.NextDouble();
        }
    }
    return matrix;
}

static float[] GenerateRandomImage(int width, int height)
{
    var random = new Random();
    var pixels = new float[width * height * 3]; // RGB
    for (int i = 0; i < pixels.Length; i++)
    {
        pixels[i] = (float)random.NextDouble();
    }
    return pixels;
}

static bool VerifyVectorAddition(float[] a, float[] b, float[] result)
{
    for (int i = 0; i < Math.Min(100, a.Length); i++)
    {
        if (Math.Abs(result[i] - (a[i] + b[i])) > 0.0001f)
        {
            return false;
        }
    }
    return true;
}

static async Task BenchmarkVectorOps(IGrainFactory grainFactory)
{
    var grain = grainFactory.GetGrain<IVectorComputeGrain>(0);
    var sizes = new[] { 1000, 10_000, 100_000, 1_000_000 };
    
    foreach (var size in sizes)
    {
        var a = GenerateRandomArray(size);
        var b = GenerateRandomArray(size);
        
        var sw = Stopwatch.StartNew();
        await grain.AddVectorsAsync(a, b);
        sw.Stop();
        
        Console.WriteLine($"   {size,10:N0} elements: {sw.ElapsedMilliseconds,6} ms");
    }
}

static async Task BenchmarkMatrixOps(IGrainFactory grainFactory)
{
    var grain = grainFactory.GetGrain<IMatrixComputeGrain>(0);
    var sizes = new[] { 64, 128, 256, 512 };
    
    foreach (var size in sizes)
    {
        var a = GenerateRandomMatrix(size);
        var b = GenerateRandomMatrix(size);
        
        var sw = Stopwatch.StartNew();
        await grain.MultiplyMatricesAsync(a, b);
        sw.Stop();
        
        var gflops = (2.0 * size * size * size) / (sw.Elapsed.TotalSeconds * 1_000_000_000);
        Console.WriteLine($"   {size,4}x{size,-4}: {sw.ElapsedMilliseconds,6} ms ({gflops:F2} GFLOPS)");
    }
}

static async Task BenchmarkReductions(IGrainFactory grainFactory)
{
    var grain = grainFactory.GetGrain<IReductionGrain>(0);
    var sizes = new[] { 10_000, 100_000, 1_000_000, 10_000_000 };
    
    foreach (var size in sizes)
    {
        var data = GenerateRandomArray(size);
        
        var sw = Stopwatch.StartNew();
        var sum = await grain.SumAsync(data);
        sw.Stop();
        
        Console.WriteLine($"   {size,10:N0} elements: {sw.ElapsedMilliseconds,6} ms");
    }
}

// Grain interfaces
public interface IVectorComputeGrain : IGrainWithIntegerKey
{
    Task<float[]> AddVectorsAsync(float[] a, float[] b);
    Task<float[]> MultiplyVectorsAsync(float[] a, float[] b);
}

public interface IMatrixComputeGrain : IGrainWithIntegerKey
{
    Task<float[,]> MultiplyMatricesAsync(float[,] a, float[,] b);
    Task<float[,]> TransposeAsync(float[,] matrix);
}

public interface IImageProcessingGrain : IGrainWithIntegerKey
{
    Task<float[]> ApplyGaussianBlurAsync(float[] pixels, int width, int height);
    Task<float[]> DetectEdgesAsync(float[] pixels, int width, int height);
    Task<float[]> SharpenImageAsync(float[] pixels, int width, int height);
}

public interface IReductionGrain : IGrainWithIntegerKey
{
    Task<float> SumAsync(float[] data);
    Task<float> MaxAsync(float[] data);
    Task<float> MinAsync(float[] data);
    Task<float> AverageAsync(float[] data);
}