using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions;
using Orleans.GpuBridge.Grains;
using Spectre.Console;

namespace Orleans.GpuBridge.Samples.VectorOperations;

public class VectorOperationsSample
{
    private readonly IServiceProvider _services;
    private readonly IGpuBridge _gpuBridge;
    
    public VectorOperationsSample(IServiceProvider services)
    {
        _services = services;
        _gpuBridge = services.GetRequiredService<IGpuBridge>();
    }
    
    public async Task RunAsync(int vectorSize, int batchSize)
    {
        AnsiConsole.MarkupLine($"[bold]Vector Operations Sample[/]");
        AnsiConsole.MarkupLine($"Vector Size: [cyan]{vectorSize:N0}[/] elements");
        AnsiConsole.MarkupLine($"Batch Size: [cyan]{batchSize}[/] operations");
        AnsiConsole.WriteLine();
        
        // Check GPU availability
        var devices = await _gpuBridge.GetAvailableDevicesAsync();
        if (devices.Count > 0)
        {
            AnsiConsole.MarkupLine($"[green]✓[/] GPU Available: [cyan]{devices[0].Name}[/]");
            AnsiConsole.MarkupLine($"  Memory: [cyan]{devices[0].TotalMemoryBytes / (1024 * 1024 * 1024):F1} GB[/]");
        }
        else
        {
            AnsiConsole.MarkupLine("[yellow]⚠[/] No GPU available, using CPU fallback");
        }
        AnsiConsole.WriteLine();
        
        // Run different vector operations
        await RunVectorAddAsync(vectorSize, batchSize);
        await RunVectorDotProductAsync(vectorSize, batchSize);
        await RunVectorNormalizeAsync(vectorSize, batchSize);
        await RunVectorReductionAsync(vectorSize, batchSize);
        
        // Show comparison
        await ShowPerformanceComparisonAsync(vectorSize);
    }
    
    public async Task RunInteractiveAsync()
    {
        var vectorSize = AnsiConsole.Ask<int>("Enter vector size:", 1000000);
        var batchSize = AnsiConsole.Ask<int>("Enter batch size:", 10);
        
        await RunAsync(vectorSize, batchSize);
    }
    
    private async Task RunVectorAddAsync(int vectorSize, int batchSize)
    {
        AnsiConsole.MarkupLine("[underline]Vector Addition[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        // Generate test data
        var vectorPairs = GenerateVectorPairs(vectorSize, batchSize);
        
        // Execute on GPU/CPU
        var sw = Stopwatch.StartNew();
        var results = new List<float[]>();
        
        foreach (var pair in vectorPairs)
        {
            var result = await _gpuBridge.ExecuteAsync<VectorPair, float[]>(
                "vector/add",
                pair);
            results.Add(result);
        }
        
        sw.Stop();
        
        // Calculate metrics
        var totalElements = (long)vectorSize * batchSize;
        var throughput = totalElements / sw.Elapsed.TotalSeconds;
        var latency = sw.Elapsed.TotalMilliseconds / batchSize;
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Throughput", $"{throughput:N0} elements/sec");
        table.AddRow("Avg Latency", $"{latency:F2} ms/operation");
        table.AddRow("Operations", $"{batchSize:N0}");
        table.AddRow("Elements", $"{totalElements:N0}");
        
        // Verify correctness
        var verified = VerifyVectorAdd(vectorPairs[0], results[0]);
        table.AddRow("Verification", verified ? "[green]✓ Passed[/]" : "[red]✗ Failed[/]");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunVectorDotProductAsync(int vectorSize, int batchSize)
    {
        AnsiConsole.MarkupLine("[underline]Vector Dot Product[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var vectorPairs = GenerateVectorPairs(vectorSize, batchSize);
        
        var sw = Stopwatch.StartNew();
        var results = new List<float>();
        
        foreach (var pair in vectorPairs)
        {
            var result = await _gpuBridge.ExecuteAsync<VectorPair, float>(
                "vector/dot",
                pair);
            results.Add(result);
        }
        
        sw.Stop();
        
        var throughput = (long)vectorSize * batchSize / sw.Elapsed.TotalSeconds;
        var latency = sw.Elapsed.TotalMilliseconds / batchSize;
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Throughput", $"{throughput:N0} elements/sec");
        table.AddRow("Avg Latency", $"{latency:F2} ms/operation");
        table.AddRow("Operations", $"{batchSize:N0}");
        
        // Verify correctness
        var verified = VerifyDotProduct(vectorPairs[0], results[0]);
        table.AddRow("Verification", verified ? "[green]✓ Passed[/]" : "[red]✗ Failed[/]");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunVectorNormalizeAsync(int vectorSize, int batchSize)
    {
        AnsiConsole.MarkupLine("[underline]Vector Normalization[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var vectors = GenerateVectors(vectorSize, batchSize);
        
        var sw = Stopwatch.StartNew();
        var results = new List<float[]>();
        
        foreach (var vector in vectors)
        {
            var result = await _gpuBridge.ExecuteAsync<float[], float[]>(
                "vector/normalize",
                vector);
            results.Add(result);
        }
        
        sw.Stop();
        
        var throughput = (long)vectorSize * batchSize / sw.Elapsed.TotalSeconds;
        var latency = sw.Elapsed.TotalMilliseconds / batchSize;
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Throughput", $"{throughput:N0} elements/sec");
        table.AddRow("Avg Latency", $"{latency:F2} ms/operation");
        table.AddRow("Operations", $"{batchSize:N0}");
        
        // Verify normalization
        var verified = VerifyNormalization(results[0]);
        table.AddRow("Verification", verified ? "[green]✓ Passed[/]" : "[red]✗ Failed[/]");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunVectorReductionAsync(int vectorSize, int batchSize)
    {
        AnsiConsole.MarkupLine("[underline]Vector Reduction (Sum)[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var vectors = GenerateVectors(vectorSize, batchSize);
        
        var sw = Stopwatch.StartNew();
        var results = new List<float>();
        
        foreach (var vector in vectors)
        {
            var result = await _gpuBridge.ExecuteAsync<float[], float>(
                "vector/sum",
                vector);
            results.Add(result);
        }
        
        sw.Stop();
        
        var throughput = (long)vectorSize * batchSize / sw.Elapsed.TotalSeconds;
        var latency = sw.Elapsed.TotalMilliseconds / batchSize;
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Throughput", $"{throughput:N0} elements/sec");
        table.AddRow("Avg Latency", $"{latency:F2} ms/operation");
        table.AddRow("Operations", $"{batchSize:N0}");
        
        // Verify sum
        var verified = VerifySum(vectors[0], results[0]);
        table.AddRow("Verification", verified ? "[green]✓ Passed[/]" : "[red]✗ Failed[/]");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task ShowPerformanceComparisonAsync(int vectorSize)
    {
        AnsiConsole.MarkupLine("[bold underline]CPU vs GPU Performance Comparison[/]");
        
        var chart = new BarChart()
            .Width(60)
            .Label("[bold]Operations per Second[/]");
        
        // Test with CPU
        var cpuOps = await BenchmarkOperationsAsync(vectorSize, false);
        chart.AddItem("CPU", cpuOps, Color.Blue);
        
        // Test with GPU (if available)
        var devices = await _gpuBridge.GetAvailableDevicesAsync();
        if (devices.Count > 0)
        {
            var gpuOps = await BenchmarkOperationsAsync(vectorSize, true);
            chart.AddItem("GPU", gpuOps, Color.Green);
            
            var speedup = gpuOps / cpuOps;
            chart.AddItem($"Speedup: {speedup:F1}x", 0, Color.Yellow);
        }
        
        AnsiConsole.Write(chart);
        AnsiConsole.WriteLine();
    }
    
    private async Task<double> BenchmarkOperationsAsync(int vectorSize, bool useGpu)
    {
        var testDuration = TimeSpan.FromSeconds(5);
        var operations = 0;
        var sw = Stopwatch.StartNew();
        
        var vector1 = GenerateRandomVector(vectorSize);
        var vector2 = GenerateRandomVector(vectorSize);
        var pair = new VectorPair(vector1, vector2);
        
        while (sw.Elapsed < testDuration)
        {
            await _gpuBridge.ExecuteAsync<VectorPair, float[]>(
                "vector/add",
                pair,
                new GpuExecutionHints { PreferGpu = useGpu });
            operations++;
        }
        
        return operations / sw.Elapsed.TotalSeconds;
    }
    
    private List<VectorPair> GenerateVectorPairs(int size, int count)
    {
        var pairs = new List<VectorPair>();
        var random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < count; i++)
        {
            var a = GenerateRandomVector(size, random);
            var b = GenerateRandomVector(size, random);
            pairs.Add(new VectorPair(a, b));
        }
        
        return pairs;
    }
    
    private List<float[]> GenerateVectors(int size, int count)
    {
        var vectors = new List<float[]>();
        var random = new Random(42);
        
        for (int i = 0; i < count; i++)
        {
            vectors.Add(GenerateRandomVector(size, random));
        }
        
        return vectors;
    }
    
    private float[] GenerateRandomVector(int size, Random? random = null)
    {
        random ??= new Random();
        var vector = new float[size];
        
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)(random.NextDouble() * 100);
        }
        
        return vector;
    }
    
    private bool VerifyVectorAdd(VectorPair input, float[] result)
    {
        if (result.Length != input.A.Length)
            return false;
        
        const float epsilon = 0.0001f;
        for (int i = 0; i < Math.Min(100, result.Length); i++) // Check first 100 elements
        {
            var expected = input.A[i] + input.B[i];
            if (Math.Abs(result[i] - expected) > epsilon)
                return false;
        }
        
        return true;
    }
    
    private bool VerifyDotProduct(VectorPair input, float result)
    {
        var expected = 0f;
        for (int i = 0; i < Math.Min(1000, input.A.Length); i++) // Check first 1000 elements
        {
            expected += input.A[i] * input.B[i];
        }
        
        // Scale expected if we only checked partial
        if (input.A.Length > 1000)
        {
            expected = expected * input.A.Length / 1000f;
        }
        
        const float tolerance = 0.01f; // 1% tolerance due to floating point
        return Math.Abs(result - expected) / expected < tolerance;
    }
    
    private bool VerifyNormalization(float[] result)
    {
        var magnitude = 0f;
        for (int i = 0; i < Math.Min(1000, result.Length); i++)
        {
            magnitude += result[i] * result[i];
        }
        magnitude = (float)Math.Sqrt(magnitude);
        
        // Should be close to 1.0 for normalized vector
        return Math.Abs(magnitude - 1.0f) < 0.01f;
    }
    
    private bool VerifySum(float[] input, float result)
    {
        var expected = input.Take(Math.Min(1000, input.Length)).Sum();
        
        // Scale if partial
        if (input.Length > 1000)
        {
            expected = expected * input.Length / 1000f;
        }
        
        const float tolerance = 0.01f;
        return Math.Abs(result - expected) / expected < tolerance;
    }
}

public record VectorPair(float[] A, float[] B);