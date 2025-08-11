using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions;
using Spectre.Console;

namespace Orleans.GpuBridge.Samples.ImageProcessing;

public class ImageProcessingSample
{
    private readonly IServiceProvider _services;
    private readonly IGpuBridge _gpuBridge;
    
    public ImageProcessingSample(IServiceProvider services)
    {
        _services = services;
        _gpuBridge = services.GetRequiredService<IGpuBridge>();
    }
    
    public async Task RunAsync(string imagePath, string operation)
    {
        AnsiConsole.MarkupLine($"[bold]Image Processing Sample[/]");
        AnsiConsole.MarkupLine($"Image: [cyan]{Path.GetFileName(imagePath)}[/]");
        AnsiConsole.MarkupLine($"Operation: [cyan]{operation}[/]");
        AnsiConsole.WriteLine();
        
        // Load or generate test image
        var imageData = await LoadImageAsync(imagePath);
        
        switch (operation.ToLower())
        {
            case "resize":
                await ProcessResizeAsync(imageData);
                break;
            case "filter":
                await ProcessFilterAsync(imageData);
                break;
            case "convolve":
                await ProcessConvolutionAsync(imageData);
                break;
            case "all":
            default:
                await ProcessResizeAsync(imageData);
                await ProcessFilterAsync(imageData);
                await ProcessConvolutionAsync(imageData);
                break;
        }
    }
    
    public async Task RunInteractiveAsync()
    {
        var operation = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("Select image operation:")
                .AddChoices("Resize", "Filter", "Convolve", "All"));
        
        await RunAsync(GenerateTestImage(), operation);
    }
    
    private async Task ProcessResizeAsync(ImageData image)
    {
        AnsiConsole.MarkupLine("[underline]Image Resize[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var resizeParams = new ResizeParameters
        {
            OriginalWidth = image.Width,
            OriginalHeight = image.Height,
            NewWidth = image.Width / 2,
            NewHeight = image.Height / 2,
            Interpolation = "bilinear"
        };
        
        var sw = Stopwatch.StartNew();
        var result = await _gpuBridge.ExecuteAsync<(ImageData, ResizeParameters), ImageData>(
            "image/resize",
            (image, resizeParams));
        sw.Stop();
        
        var pixelsProcessed = image.Width * image.Height;
        var throughput = pixelsProcessed / sw.Elapsed.TotalSeconds / 1e6;
        
        table.AddRow("Processing Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Throughput", $"{throughput:F2} MPixels/s");
        table.AddRow("Original Size", $"{image.Width}x{image.Height}");
        table.AddRow("New Size", $"{resizeParams.NewWidth}x{resizeParams.NewHeight}");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task ProcessFilterAsync(ImageData image)
    {
        AnsiConsole.MarkupLine("[underline]Image Filtering[/]");
        
        var table = new Table();
        table.AddColumn("Filter");
        table.AddColumn(new TableColumn("Time (ms)").RightAligned());
        table.AddColumn(new TableColumn("Throughput (MP/s)").RightAligned());
        
        var filters = new[] { "gaussian", "sobel", "sharpen", "blur" };
        
        foreach (var filter in filters)
        {
            var sw = Stopwatch.StartNew();
            var result = await _gpuBridge.ExecuteAsync<(ImageData, string), ImageData>(
                "image/filter",
                (image, filter));
            sw.Stop();
            
            var throughput = image.Width * image.Height / sw.Elapsed.TotalSeconds / 1e6;
            table.AddRow(filter, $"{sw.ElapsedMilliseconds:N0}", $"{throughput:F2}");
        }
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task ProcessConvolutionAsync(ImageData image)
    {
        AnsiConsole.MarkupLine("[underline]Image Convolution[/]");
        
        var table = new Table();
        table.AddColumn("Kernel Size");
        table.AddColumn(new TableColumn("Time (ms)").RightAligned());
        table.AddColumn(new TableColumn("GFLOPS").RightAligned());
        
        var kernelSizes = new[] { 3, 5, 7, 9 };
        
        foreach (var size in kernelSizes)
        {
            var kernel = GenerateGaussianKernel(size);
            
            var sw = Stopwatch.StartNew();
            var result = await _gpuBridge.ExecuteAsync<(ImageData, float[]), ImageData>(
                "image/convolve",
                (image, kernel));
            sw.Stop();
            
            var operations = (double)image.Width * image.Height * size * size * image.Channels;
            var gflops = operations / sw.Elapsed.TotalSeconds / 1e9;
            
            table.AddRow($"{size}x{size}", $"{sw.ElapsedMilliseconds:N0}", $"{gflops:F2}");
        }
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task<ImageData> LoadImageAsync(string path)
    {
        // For demo, generate a test image
        if (!File.Exists(path))
        {
            return GenerateTestImageData();
        }
        
        // In real implementation, would load actual image
        await Task.CompletedTask;
        return GenerateTestImageData();
    }
    
    private ImageData GenerateTestImageData()
    {
        const int width = 1920;
        const int height = 1080;
        const int channels = 3; // RGB
        
        var data = new byte[width * height * channels];
        var random = new Random(42);
        random.NextBytes(data);
        
        return new ImageData
        {
            Data = data,
            Width = width,
            Height = height,
            Channels = channels,
            Format = "RGB"
        };
    }
    
    private string GenerateTestImage()
    {
        return Path.GetTempFileName() + ".bmp";
    }
    
    private float[] GenerateGaussianKernel(int size)
    {
        var kernel = new float[size * size];
        var sigma = size / 3.0f;
        var sum = 0f;
        var center = size / 2;
        
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                var dx = x - center;
                var dy = y - center;
                var value = (float)Math.Exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[y * size + x] = value;
                sum += value;
            }
        }
        
        // Normalize
        for (int i = 0; i < kernel.Length; i++)
        {
            kernel[i] /= sum;
        }
        
        return kernel;
    }
}

public class ImageData
{
    public byte[] Data { get; set; } = Array.Empty<byte>();
    public int Width { get; set; }
    public int Height { get; set; }
    public int Channels { get; set; }
    public string Format { get; set; } = "RGB";
}

public class ResizeParameters
{
    public int OriginalWidth { get; set; }
    public int OriginalHeight { get; set; }
    public int NewWidth { get; set; }
    public int NewHeight { get; set; }
    public string Interpolation { get; set; } = "bilinear";
}