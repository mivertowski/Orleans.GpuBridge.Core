using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions;
using Spectre.Console;

namespace Orleans.GpuBridge.Samples;

public class BenchmarkSample
{
    private readonly IServiceProvider _services;
    private readonly IGpuBridge _gpuBridge;
    
    public BenchmarkSample(IServiceProvider services)
    {
        _services = services;
        _gpuBridge = services.GetRequiredService<IGpuBridge>();
    }
    
    public async Task RunAsync(string type, int durationSeconds)
    {
        AnsiConsole.MarkupLine($"[bold]Performance Benchmark[/]");
        AnsiConsole.MarkupLine($"Type: [cyan]{type}[/]");
        AnsiConsole.MarkupLine($"Duration: [cyan]{durationSeconds}[/] seconds");
        AnsiConsole.WriteLine();
        
        switch (type.ToLower())
        {
            case "cpu":
                await RunCpuBenchmarkAsync(durationSeconds);
                break;
            case "gpu":
                await RunGpuBenchmarkAsync(durationSeconds);
                break;
            case "all":
            default:
                await RunCpuBenchmarkAsync(durationSeconds);
                await RunGpuBenchmarkAsync(durationSeconds);
                await RunComparisonBenchmarkAsync(durationSeconds);
                break;
        }
    }
    
    public async Task RunInteractiveAsync()
    {
        var type = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Select benchmark type:")
                .AddChoices("CPU Only", "GPU Only", "CPU vs GPU", "All"));
        
        var duration = AnsiConsole.Ask<int>("Duration in seconds:", 30);
        
        await RunAsync(type.Replace(" Only", "").Replace("CPU vs GPU", "all"), duration);
    }
    
    private async Task RunCpuBenchmarkAsync(int durationSeconds)
    {
        AnsiConsole.MarkupLine("[underline]CPU Benchmark[/]");
        
        var results = new Dictionary<string, BenchmarkResult>();
        
        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("Running CPU benchmarks", maxValue: 4);
                
                // Vector operations
                task.Description = "Vector operations...";
                results["Vector Add"] = await BenchmarkOperationAsync(
                    "vector/add",
                    GenerateVectorPair(1000000),
                    durationSeconds,
                    false);
                task.Increment(1);
                
                // Matrix operations
                task.Description = "Matrix multiply...";
                results["Matrix Multiply"] = await BenchmarkOperationAsync(
                    "matrix/multiply",
                    GenerateMatrixPair(512),
                    durationSeconds,
                    false);
                task.Increment(1);
                
                // Reduction
                task.Description = "Vector reduction...";
                results["Vector Sum"] = await BenchmarkOperationAsync(
                    "vector/sum",
                    GenerateVector(1000000),
                    durationSeconds,
                    false);
                task.Increment(1);
                
                // Memory transfer
                task.Description = "Memory operations...";
                results["Memory Copy"] = await BenchmarkMemoryAsync(
                    1024 * 1024,
                    durationSeconds,
                    false);
                task.Increment(1);
            });
        
        DisplayResults("CPU", results);
    }
    
    private async Task RunGpuBenchmarkAsync(int durationSeconds)
    {
        var devices = await _gpuBridge.GetAvailableDevicesAsync();
        if (devices.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No GPU devices available[/]");
            return;
        }
        
        AnsiConsole.MarkupLine("[underline]GPU Benchmark[/]");
        
        var results = new Dictionary<string, BenchmarkResult>();
        
        await AnsiConsole.Progress()
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask("Running GPU benchmarks", maxValue: 4);
                
                // Vector operations
                task.Description = "Vector operations...";
                results["Vector Add"] = await BenchmarkOperationAsync(
                    "vector/add",
                    GenerateVectorPair(10000000),
                    durationSeconds,
                    true);
                task.Increment(1);
                
                // Matrix operations
                task.Description = "Matrix multiply...";
                results["Matrix Multiply"] = await BenchmarkOperationAsync(
                    "matrix/multiply",
                    GenerateMatrixPair(2048),
                    durationSeconds,
                    true);
                task.Increment(1);
                
                // Reduction
                task.Description = "Vector reduction...";
                results["Vector Sum"] = await BenchmarkOperationAsync(
                    "vector/sum",
                    GenerateVector(10000000),
                    durationSeconds,
                    true);
                task.Increment(1);
                
                // Memory transfer
                task.Description = "Memory operations...";
                results["Memory Copy"] = await BenchmarkMemoryAsync(
                    128 * 1024 * 1024,
                    durationSeconds,
                    true);
                task.Increment(1);
            });
        
        DisplayResults("GPU", results);
    }
    
    private async Task RunComparisonBenchmarkAsync(int durationSeconds)
    {
        AnsiConsole.MarkupLine("[underline]CPU vs GPU Comparison[/]");
        
        var testSizes = new[] { 1000, 10000, 100000, 1000000, 10000000 };
        var cpuResults = new List<double>();
        var gpuResults = new List<double>();
        
        var chart = new BarChart()
            .Width(60)
            .Label("[bold]Operations per Second[/]");
        
        foreach (var size in testSizes)
        {
            // CPU test
            var cpuOps = await QuickBenchmarkAsync(
                "vector/add",
                GenerateVectorPair(size),
                TimeSpan.FromSeconds(2),
                false);
            cpuResults.Add(cpuOps);
            
            // GPU test
            var gpuOps = await QuickBenchmarkAsync(
                "vector/add",
                GenerateVectorPair(size),
                TimeSpan.FromSeconds(2),
                true);
            gpuResults.Add(gpuOps);
            
            var speedup = gpuOps / cpuOps;
            chart.AddItem($"Size {size:N0}", speedup, 
                speedup > 1 ? Color.Green : Color.Red);
        }
        
        AnsiConsole.Write(chart);
        AnsiConsole.WriteLine();
        
        // Summary table
        var table = new Table();
        table.AddColumn("Size");
        table.AddColumn(new TableColumn("CPU Ops/s").RightAligned());
        table.AddColumn(new TableColumn("GPU Ops/s").RightAligned());
        table.AddColumn(new TableColumn("Speedup").RightAligned());
        
        for (int i = 0; i < testSizes.Length; i++)
        {
            var speedup = gpuResults[i] / cpuResults[i];
            table.AddRow(
                $"{testSizes[i]:N0}",
                $"{cpuResults[i]:N0}",
                $"{gpuResults[i]:N0}",
                $"{speedup:F2}x");
        }
        
        AnsiConsole.Write(table);
    }
    
    private async Task<BenchmarkResult> BenchmarkOperationAsync<T>(
        string kernelId,
        T input,
        int durationSeconds,
        bool useGpu)
    {
        var duration = TimeSpan.FromSeconds(durationSeconds);
        var operations = 0;
        var totalLatency = 0.0;
        var sw = Stopwatch.StartNew();
        
        while (sw.Elapsed < duration)
        {
            var opSw = Stopwatch.StartNew();
            await _gpuBridge.ExecuteAsync<T, object>(
                kernelId,
                input,
                new GpuExecutionHints { PreferGpu = useGpu });
            opSw.Stop();
            
            operations++;
            totalLatency += opSw.Elapsed.TotalMilliseconds;
        }
        
        sw.Stop();
        
        return new BenchmarkResult
        {
            Operations = operations,
            TotalTime = sw.Elapsed,
            AverageLatency = totalLatency / operations,
            Throughput = operations / sw.Elapsed.TotalSeconds
        };
    }
    
    private async Task<BenchmarkResult> BenchmarkMemoryAsync(
        int sizeBytes,
        int durationSeconds,
        bool useGpu)
    {
        var duration = TimeSpan.FromSeconds(durationSeconds);
        var operations = 0;
        var sw = Stopwatch.StartNew();
        var data = new byte[sizeBytes];
        
        while (sw.Elapsed < duration)
        {
            await _gpuBridge.ExecuteAsync<byte[], byte[]>(
                "memory/copy",
                data,
                new GpuExecutionHints { PreferGpu = useGpu });
            operations++;
        }
        
        sw.Stop();
        
        var totalBytes = (long)operations * sizeBytes;
        var bandwidth = totalBytes / sw.Elapsed.TotalSeconds / 1e9;
        
        return new BenchmarkResult
        {
            Operations = operations,
            TotalTime = sw.Elapsed,
            AverageLatency = sw.Elapsed.TotalMilliseconds / operations,
            Throughput = bandwidth // GB/s for memory
        };
    }
    
    private async Task<double> QuickBenchmarkAsync<T>(
        string kernelId,
        T input,
        TimeSpan duration,
        bool useGpu)
    {
        var operations = 0;
        var sw = Stopwatch.StartNew();
        
        while (sw.Elapsed < duration)
        {
            await _gpuBridge.ExecuteAsync<T, object>(
                kernelId,
                input,
                new GpuExecutionHints { PreferGpu = useGpu });
            operations++;
        }
        
        return operations / sw.Elapsed.TotalSeconds;
    }
    
    private void DisplayResults(string type, Dictionary<string, BenchmarkResult> results)
    {
        var table = new Table();
        table.Title = new TableTitle($"{type} Benchmark Results");
        table.AddColumn("Operation");
        table.AddColumn(new TableColumn("Operations").RightAligned());
        table.AddColumn(new TableColumn("Throughput").RightAligned());
        table.AddColumn(new TableColumn("Avg Latency").RightAligned());
        
        foreach (var (operation, result) in results)
        {
            var throughputStr = operation.Contains("Memory") 
                ? $"{result.Throughput:F2} GB/s"
                : $"{result.Throughput:N0} ops/s";
            
            table.AddRow(
                operation,
                $"{result.Operations:N0}",
                throughputStr,
                $"{result.AverageLatency:F2} ms");
        }
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private object GenerateVectorPair(int size)
    {
        var a = new float[size];
        var b = new float[size];
        var random = new Random(42);
        
        for (int i = 0; i < size; i++)
        {
            a[i] = (float)random.NextDouble();
            b[i] = (float)random.NextDouble();
        }
        
        return new { A = a, B = b };
    }
    
    private object GenerateMatrixPair(int size)
    {
        var a = new float[size * size];
        var b = new float[size * size];
        var random = new Random(42);
        
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)random.NextDouble();
            b[i] = (float)random.NextDouble();
        }
        
        return new { A = a, B = b, Size = size };
    }
    
    private float[] GenerateVector(int size)
    {
        var vector = new float[size];
        var random = new Random(42);
        
        for (int i = 0; i < size; i++)
        {
            vector[i] = (float)random.NextDouble();
        }
        
        return vector;
    }
}

public class BenchmarkResult
{
    public int Operations { get; set; }
    public TimeSpan TotalTime { get; set; }
    public double AverageLatency { get; set; }
    public double Throughput { get; set; }
}