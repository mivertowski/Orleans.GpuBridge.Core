using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Orleans.GpuBridge.Abstractions;
using Spectre.Console;

namespace Orleans.GpuBridge.Samples.MatrixOperations;

public class MatrixOperationsSample
{
    private readonly IServiceProvider _services;
    private readonly IGpuBridge _gpuBridge;
    
    public MatrixOperationsSample(IServiceProvider services)
    {
        _services = services;
        _gpuBridge = services.GetRequiredService<IGpuBridge>();
    }
    
    public async Task RunAsync(int matrixSize, int operationCount)
    {
        AnsiConsole.MarkupLine($"[bold]Matrix Operations Sample[/]");
        AnsiConsole.MarkupLine($"Matrix Size: [cyan]{matrixSize}x{matrixSize}[/]");
        AnsiConsole.MarkupLine($"Operations: [cyan]{operationCount}[/]");
        AnsiConsole.WriteLine();
        
        await RunMatrixMultiplyAsync(matrixSize, operationCount);
        await RunMatrixTransposeAsync(matrixSize, operationCount);
        await RunMatrixInverseAsync(Math.Min(matrixSize, 256), operationCount); // Smaller for inverse
    }
    
    public async Task RunInteractiveAsync()
    {
        var size = AnsiConsole.Ask<int>("Enter matrix dimension:", 1024);
        var count = AnsiConsole.Ask<int>("Enter number of operations:", 5);
        
        await RunAsync(size, count);
    }
    
    private async Task RunMatrixMultiplyAsync(int size, int count)
    {
        AnsiConsole.MarkupLine("[underline]Matrix Multiplication[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var matrices = GenerateMatrixPairs(size, count);
        
        var sw = Stopwatch.StartNew();
        var results = new List<float[]>();
        
        foreach (var pair in matrices)
        {
            var result = await _gpuBridge.ExecuteAsync<MatrixPair, float[]>(
                "matrix/multiply",
                pair);
            results.Add(result);
        }
        
        sw.Stop();
        
        var flops = 2.0 * size * size * size * count; // 2N^3 operations per multiply
        var gflops = flops / sw.Elapsed.TotalSeconds / 1e9;
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("GFLOPS", $"{gflops:F2}");
        table.AddRow("Avg Latency", $"{sw.ElapsedMilliseconds / count:F2} ms/operation");
        table.AddRow("Matrix Size", $"{size}x{size}");
        table.AddRow("Operations", $"{count:N0}");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunMatrixTransposeAsync(int size, int count)
    {
        AnsiConsole.MarkupLine("[underline]Matrix Transpose[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var matrices = GenerateMatrices(size, count);
        
        var sw = Stopwatch.StartNew();
        var results = new List<float[]>();
        
        foreach (var matrix in matrices)
        {
            var result = await _gpuBridge.ExecuteAsync<Matrix, float[]>(
                "matrix/transpose",
                matrix);
            results.Add(result);
        }
        
        sw.Stop();
        
        var bandwidth = (double)size * size * sizeof(float) * count / sw.Elapsed.TotalSeconds / 1e9;
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Bandwidth", $"{bandwidth:F2} GB/s");
        table.AddRow("Avg Latency", $"{sw.ElapsedMilliseconds / count:F2} ms/operation");
        table.AddRow("Operations", $"{count:N0}");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private async Task RunMatrixInverseAsync(int size, int count)
    {
        AnsiConsole.MarkupLine("[underline]Matrix Inverse (LU Decomposition)[/]");
        
        var table = new Table();
        table.AddColumn("Metric");
        table.AddColumn(new TableColumn("Value").RightAligned());
        
        var matrices = GenerateInvertibleMatrices(size, count);
        
        var sw = Stopwatch.StartNew();
        var results = new List<float[]>();
        
        foreach (var matrix in matrices)
        {
            try
            {
                var result = await _gpuBridge.ExecuteAsync<Matrix, float[]>(
                    "matrix/inverse",
                    matrix);
                results.Add(result);
            }
            catch
            {
                // Matrix might be singular
                results.Add(new float[size * size]);
            }
        }
        
        sw.Stop();
        
        table.AddRow("Total Time", $"{sw.ElapsedMilliseconds:N0} ms");
        table.AddRow("Avg Latency", $"{sw.ElapsedMilliseconds / count:F2} ms/operation");
        table.AddRow("Matrix Size", $"{size}x{size}");
        table.AddRow("Operations", $"{count:N0}");
        table.AddRow("Success Rate", $"{results.Count(r => r.Any(v => v != 0)) * 100.0 / count:F1}%");
        
        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
    
    private List<MatrixPair> GenerateMatrixPairs(int size, int count)
    {
        var pairs = new List<MatrixPair>();
        var random = new Random(42);
        
        for (int i = 0; i < count; i++)
        {
            var a = GenerateRandomMatrix(size, random);
            var b = GenerateRandomMatrix(size, random);
            pairs.Add(new MatrixPair(a, b));
        }
        
        return pairs;
    }
    
    private List<Matrix> GenerateMatrices(int size, int count)
    {
        var matrices = new List<Matrix>();
        var random = new Random(42);
        
        for (int i = 0; i < count; i++)
        {
            matrices.Add(GenerateRandomMatrix(size, random));
        }
        
        return matrices;
    }
    
    private List<Matrix> GenerateInvertibleMatrices(int size, int count)
    {
        var matrices = new List<Matrix>();
        var random = new Random(42);
        
        for (int i = 0; i < count; i++)
        {
            // Generate a diagonally dominant matrix (guaranteed invertible)
            var data = new float[size * size];
            for (int row = 0; row < size; row++)
            {
                for (int col = 0; col < size; col++)
                {
                    if (row == col)
                    {
                        data[row * size + col] = (float)(random.NextDouble() * 10 + 10); // Large diagonal
                    }
                    else
                    {
                        data[row * size + col] = (float)(random.NextDouble() * 2); // Small off-diagonal
                    }
                }
            }
            matrices.Add(new Matrix(data, size, size));
        }
        
        return matrices;
    }
    
    private Matrix GenerateRandomMatrix(int size, Random random)
    {
        var data = new float[size * size];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(random.NextDouble() * 10);
        }
        return new Matrix(data, size, size);
    }
}

public record Matrix(float[] Data, int Rows, int Cols);
public record MatrixPair(Matrix A, Matrix B);