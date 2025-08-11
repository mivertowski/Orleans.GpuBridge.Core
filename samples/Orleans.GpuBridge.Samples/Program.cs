using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Samples;
using Orleans.GpuBridge.Samples.VectorOperations;
using Orleans.GpuBridge.Samples.MatrixOperations;
using Orleans.GpuBridge.Samples.ImageProcessing;
using Orleans.GpuBridge.Samples.GraphProcessing;
using Spectre.Console;

var rootCommand = new RootCommand("Orleans.GpuBridge Sample Applications");

// Vector operations command
var vectorCommand = new Command("vector", "Run vector operation samples");
var vectorAddOption = new Option<int>("--size", () => 1000000, "Vector size");
var vectorBatchOption = new Option<int>("--batch", () => 10, "Batch size");
vectorCommand.AddOption(vectorAddOption);
vectorCommand.AddOption(vectorBatchOption);
vectorCommand.SetHandler(async (size, batch) =>
{
    await RunSampleAsync("Vector Operations", async (host) =>
    {
        var sample = new VectorOperationsSample(host.Services);
        await sample.RunAsync(size, batch);
    });
}, vectorAddOption, vectorBatchOption);

// Matrix operations command
var matrixCommand = new Command("matrix", "Run matrix operation samples");
var matrixSizeOption = new Option<int>("--size", () => 1024, "Matrix dimension");
var matrixCountOption = new Option<int>("--count", () => 5, "Number of operations");
matrixCommand.AddOption(matrixSizeOption);
matrixCommand.AddOption(matrixCountOption);
matrixCommand.SetHandler(async (size, count) =>
{
    await RunSampleAsync("Matrix Operations", async (host) =>
    {
        var sample = new MatrixOperationsSample(host.Services);
        await sample.RunAsync(size, count);
    });
}, matrixSizeOption, matrixCountOption);

// Image processing command
var imageCommand = new Command("image", "Run image processing samples");
var imagePathOption = new Option<string>("--path", "Path to image file");
var imageOperationOption = new Option<string>("--operation", () => "all", "Operation: resize|filter|convolve|all");
imageCommand.AddOption(imagePathOption);
imageCommand.AddOption(imageOperationOption);
imageCommand.SetHandler(async (path, operation) =>
{
    await RunSampleAsync("Image Processing", async (host) =>
    {
        var sample = new ImageProcessingSample(host.Services);
        await sample.RunAsync(path ?? GenerateTestImage(), operation);
    });
}, imagePathOption, imageOperationOption);

// Graph processing command
var graphCommand = new Command("graph", "Run graph processing samples");
var graphNodesOption = new Option<int>("--nodes", () => 10000, "Number of nodes");
var graphEdgesOption = new Option<int>("--edges", () => 50000, "Number of edges");
var graphAlgorithmOption = new Option<string>("--algorithm", () => "pagerank", "Algorithm: pagerank|shortest|traversal|all");
graphCommand.AddOption(graphNodesOption);
graphCommand.AddOption(graphEdgesOption);
graphCommand.AddOption(graphAlgorithmOption);
graphCommand.SetHandler(async (nodes, edges, algorithm) =>
{
    await RunSampleAsync("Graph Processing", async (host) =>
    {
        var sample = new GraphProcessingSample(host.Services);
        await sample.RunAsync(nodes, edges, algorithm);
    });
}, graphNodesOption, graphEdgesOption, graphAlgorithmOption);

// Benchmark command
var benchmarkCommand = new Command("benchmark", "Run performance benchmarks");
var benchmarkTypeOption = new Option<string>("--type", () => "all", "Benchmark type: cpu|gpu|all");
var benchmarkDurationOption = new Option<int>("--duration", () => 30, "Duration in seconds");
benchmarkCommand.AddOption(benchmarkTypeOption);
benchmarkCommand.AddOption(benchmarkDurationOption);
benchmarkCommand.SetHandler(async (type, duration) =>
{
    await RunSampleAsync("Performance Benchmark", async (host) =>
    {
        var sample = new BenchmarkSample(host.Services);
        await sample.RunAsync(type, duration);
    });
}, benchmarkTypeOption, benchmarkDurationOption);

// Interactive command
var interactiveCommand = new Command("interactive", "Run interactive sample selector");
interactiveCommand.SetHandler(async () =>
{
    await RunInteractiveSampleAsync();
});

rootCommand.AddCommand(vectorCommand);
rootCommand.AddCommand(matrixCommand);
rootCommand.AddCommand(imageCommand);
rootCommand.AddCommand(graphCommand);
rootCommand.AddCommand(benchmarkCommand);
rootCommand.AddCommand(interactiveCommand);

// Add global options
var verboseOption = new Option<bool>("--verbose", "Enable verbose logging");
var gpuOption = new Option<bool>("--gpu", () => true, "Use GPU acceleration");
rootCommand.AddGlobalOption(verboseOption);
rootCommand.AddGlobalOption(gpuOption);

return await rootCommand.InvokeAsync(args);

async Task RunSampleAsync(string sampleName, Func<IHost, Task> runAction)
{
    AnsiConsole.Write(new FigletText("Orleans.GpuBridge")
        .Centered()
        .Color(Color.Green));
    
    AnsiConsole.WriteLine();
    AnsiConsole.MarkupLine($"[bold yellow]Running {sampleName} Sample[/]");
    AnsiConsole.WriteLine();
    
    await AnsiConsole.Status()
        .Spinner(Spinner.Known.Star)
        .StartAsync("Initializing Orleans cluster...", async ctx =>
        {
            var host = CreateHost();
            await host.StartAsync();
            
            ctx.Status("Running sample...");
            
            try
            {
                await runAction(host);
            }
            catch (Exception ex)
            {
                AnsiConsole.WriteException(ex);
            }
            finally
            {
                await host.StopAsync();
            }
        });
}

async Task RunInteractiveSampleAsync()
{
    AnsiConsole.Write(new FigletText("Orleans.GpuBridge")
        .Centered()
        .Color(Color.Green));
    
    while (true)
    {
        var choice = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("[yellow]Select a sample to run:[/]")
                .AddChoices(new[]
                {
                    "Vector Operations",
                    "Matrix Operations",
                    "Image Processing",
                    "Graph Processing",
                    "Performance Benchmark",
                    "Exit"
                }));
        
        if (choice == "Exit")
            break;
        
        var host = CreateHost();
        await host.StartAsync();
        
        try
        {
            switch (choice)
            {
                case "Vector Operations":
                    var vectorSample = new VectorOperationsSample(host.Services);
                    await vectorSample.RunInteractiveAsync();
                    break;
                case "Matrix Operations":
                    var matrixSample = new MatrixOperationsSample(host.Services);
                    await matrixSample.RunInteractiveAsync();
                    break;
                case "Image Processing":
                    var imageSample = new ImageProcessingSample(host.Services);
                    await imageSample.RunInteractiveAsync();
                    break;
                case "Graph Processing":
                    var graphSample = new GraphProcessingSample(host.Services);
                    await graphSample.RunInteractiveAsync();
                    break;
                case "Performance Benchmark":
                    var benchmarkSample = new BenchmarkSample(host.Services);
                    await benchmarkSample.RunInteractiveAsync();
                    break;
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.WriteException(ex);
        }
        finally
        {
            await host.StopAsync();
        }
        
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[dim]Press any key to continue...[/]");
        Console.ReadKey(true);
        AnsiConsole.Clear();
    }
}

IHost CreateHost()
{
    return Host.CreateDefaultBuilder()
        .UseOrleans(builder =>
        {
            builder
                .UseLocalhostClustering()
                .AddMemoryGrainStorage("gpu-storage");
        })
        .ConfigureServices((context, services) =>
        {
            services.AddGpuBridge(options =>
            {
                options.PreferGpu = true;
                options.EnableCpuFallback = true;
            });
            
            services.AddGpuTelemetry(telemetry =>
            {
                telemetry.EnableConsoleExporter = true;
            });
        })
        .ConfigureLogging(logging =>
        {
            logging.ClearProviders();
            logging.AddConsole();
            logging.SetMinimumLevel(LogLevel.Warning);
        })
        .Build();
}

string GenerateTestImage()
{
    // Generate a simple test image
    var tempPath = Path.GetTempFileName() + ".bmp";
    // Implementation would create a simple bitmap
    return tempPath;
}