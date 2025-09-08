using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Providers.Execution.Results;
using Orleans.GpuBridge.Backends.ILGPU.Kernels.Templates;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// Registry for managing and executing pre-compiled kernel templates
/// </summary>
public sealed class KernelTemplateRegistry
{
    private readonly ILogger<KernelTemplateRegistry> _logger;
    private readonly Context _context;
    private readonly Dictionary<string, KernelTemplate> _templates;
    private readonly Dictionary<string, CompiledKernelInfo> _compiledKernels;
    private readonly SemaphoreSlim _compilationLock;

    public KernelTemplateRegistry(ILogger<KernelTemplateRegistry> logger, Context context)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _templates = new Dictionary<string, KernelTemplate>();
        _compiledKernels = new Dictionary<string, CompiledKernelInfo>();
        _compilationLock = new SemaphoreSlim(1, 1);
        
        RegisterBuiltInTemplates();
    }

    /// <summary>
    /// Registers all built-in kernel templates
    /// </summary>
    private void RegisterBuiltInTemplates()
    {
        // Vector operations
        RegisterTemplate("vector/add", typeof(VectorOperationKernels), nameof(VectorOperationKernels.VectorAdd));
        RegisterTemplate("vector/subtract", typeof(VectorOperationKernels), nameof(VectorOperationKernels.VectorSubtract));
        RegisterTemplate("vector/multiply", typeof(VectorOperationKernels), nameof(VectorOperationKernels.VectorMultiply));
        RegisterTemplate("vector/divide", typeof(VectorOperationKernels), nameof(VectorOperationKernels.VectorDivide));
        RegisterTemplate("vector/scalar_multiply", typeof(VectorOperationKernels), nameof(VectorOperationKernels.ScalarMultiply));
        RegisterTemplate("vector/scalar_add", typeof(VectorOperationKernels), nameof(VectorOperationKernels.ScalarAdd));
        RegisterTemplate("vector/fma", typeof(VectorOperationKernels), nameof(VectorOperationKernels.FusedMultiplyAdd));
        RegisterTemplate("vector/dot_partial", typeof(VectorOperationKernels), nameof(VectorOperationKernels.DotProductPartial));
        RegisterTemplate("vector/norm_partial", typeof(VectorOperationKernels), nameof(VectorOperationKernels.VectorNormPartial));
        RegisterTemplate("vector/clamp", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Clamp));
        RegisterTemplate("vector/abs", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Abs));
        RegisterTemplate("vector/power", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Power));
        RegisterTemplate("vector/sqrt", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Sqrt));
        RegisterTemplate("vector/exp", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Exp));
        RegisterTemplate("vector/log", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Log));
        RegisterTemplate("vector/sin", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Sin));
        RegisterTemplate("vector/cos", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Cos));
        RegisterTemplate("vector/saxpy", typeof(VectorOperationKernels), nameof(VectorOperationKernels.SAXPY));
        RegisterTemplate("vector/copy", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Copy));
        RegisterTemplate("vector/fill", typeof(VectorOperationKernels), nameof(VectorOperationKernels.Fill));
        
        // Reduction operations
        RegisterTemplate("reduction/sum", typeof(ReductionKernels), nameof(ReductionKernels.SumReduction));
        RegisterTemplate("reduction/max", typeof(ReductionKernels), nameof(ReductionKernels.MaxReduction));
        RegisterTemplate("reduction/min", typeof(ReductionKernels), nameof(ReductionKernels.MinReduction));
        RegisterTemplate("reduction/product", typeof(ReductionKernels), nameof(ReductionKernels.ProductReduction));
        RegisterTemplate("reduction/count_nonzero", typeof(ReductionKernels), nameof(ReductionKernels.CountNonZero));
        RegisterTemplate("reduction/average", typeof(ReductionKernels), nameof(ReductionKernels.AverageReduction));
        RegisterTemplate("reduction/stddev_pass1", typeof(ReductionKernels), nameof(ReductionKernels.StdDevPass1));
        RegisterTemplate("reduction/stddev_pass2", typeof(ReductionKernels), nameof(ReductionKernels.StdDevPass2));
        RegisterTemplate("reduction/argmax", typeof(ReductionKernels), nameof(ReductionKernels.ArgMaxReduction));
        RegisterTemplate("reduction/histogram", typeof(ReductionKernels), nameof(ReductionKernels.Histogram));
        
        // Matrix operations
        RegisterTemplate("matrix/multiply", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixMultiply));
        RegisterTemplate("matrix/transpose", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixTranspose));
        RegisterTemplate("matrix/add", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixAdd));
        RegisterTemplate("matrix/subtract", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixSubtract));
        RegisterTemplate("matrix/scalar_multiply", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixScalarMultiply));
        RegisterTemplate("matrix/elementwise_multiply", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixElementwiseMultiply));
        RegisterTemplate("matrix/vector_multiply", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.MatrixVectorMultiply));
        RegisterTemplate("matrix/outer_product", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.OuterProduct));
        RegisterTemplate("matrix/row_sum", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.RowSum));
        RegisterTemplate("matrix/column_sum", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.ColumnSum));
        RegisterTemplate("matrix/extract_diagonal", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.ExtractDiagonal));
        RegisterTemplate("matrix/set_diagonal", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.SetDiagonal));
        RegisterTemplate("matrix/normalize_rows", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.NormalizeRows));
        RegisterTemplate("matrix/relu", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.ApplyReLU));
        RegisterTemplate("matrix/sigmoid", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.ApplySigmoid));
        RegisterTemplate("matrix/tanh", typeof(MatrixOperationKernels), nameof(MatrixOperationKernels.ApplyTanh));
        
        // Image processing operations
        RegisterTemplate("image/rgb_to_grayscale", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.RGBToGrayscale));
        RegisterTemplate("image/gaussian_blur_3x3", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.GaussianBlur3x3));
        RegisterTemplate("image/sobel_edge", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.SobelEdgeDetection));
        RegisterTemplate("image/threshold", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.Threshold));
        RegisterTemplate("image/histogram_equalization", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.HistogramEqualization));
        RegisterTemplate("image/box_blur", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.BoxBlur));
        RegisterTemplate("image/rotate", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.RotateImage));
        RegisterTemplate("image/scale_nearest", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.ScaleImageNearestNeighbor));
        RegisterTemplate("image/median_filter_3x3", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.MedianFilter3x3));
        RegisterTemplate("image/brightness", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.AdjustBrightness));
        RegisterTemplate("image/contrast", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.AdjustContrast));
        RegisterTemplate("image/gamma", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.GammaCorrection));
        RegisterTemplate("image/convolution", typeof(ImageProcessingKernels), nameof(ImageProcessingKernels.Convolution));
        
        _logger.LogInformation("Registered {Count} kernel templates", _templates.Count);
    }

    /// <summary>
    /// Registers a kernel template
    /// </summary>
    private void RegisterTemplate(string name, Type type, string methodName)
    {
        var method = type.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static);
        if (method == null)
        {
            _logger.LogWarning("Failed to find kernel method: {Type}.{Method}", type.Name, methodName);
            return;
        }
        
        _templates[name] = new KernelTemplate
        {
            Name = name,
            Type = type,
            Method = method,
            Category = GetCategoryFromName(name)
        };
    }

    /// <summary>
    /// Gets a compiled kernel template for execution
    /// </summary>
    public async Task<CompiledKernelInfo?> GetCompiledKernelAsync(
        string templateName,
        Accelerator accelerator,
        CancellationToken cancellationToken = default)
    {
        if (!_templates.TryGetValue(templateName, out var template))
        {
            _logger.LogWarning("Kernel template not found: {Template}", templateName);
            return null;
        }
        
        var cacheKey = $"{templateName}_{accelerator.AcceleratorType}";
        
        // Check if already compiled
        if (_compiledKernels.TryGetValue(cacheKey, out var compiled))
        {
            return compiled;
        }
        
        // Compile the kernel
        await _compilationLock.WaitAsync(cancellationToken);
        try
        {
            // Double-check after acquiring lock
            if (_compiledKernels.TryGetValue(cacheKey, out compiled))
            {
                return compiled;
            }
            
            _logger.LogInformation("Compiling kernel template: {Template} for {Accelerator}",
                templateName, accelerator.AcceleratorType);
            
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            // Compile the kernel using ILGPU
            var kernel = accelerator.LoadAutoGroupedStreamKernel(template.Method);
            
            stopwatch.Stop();
            
            compiled = new CompiledKernelInfo
            {
                Template = template,
                Kernel = kernel,
                Accelerator = accelerator,
                CompilationTime = stopwatch.Elapsed
            };
            
            _compiledKernels[cacheKey] = compiled;
            
            _logger.LogInformation("Successfully compiled kernel template: {Template} in {Time}ms",
                templateName, stopwatch.ElapsedMilliseconds);
            
            return compiled;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to compile kernel template: {Template}", templateName);
            throw;
        }
        finally
        {
            _compilationLock.Release();
        }
    }

    /// <summary>
    /// Executes a kernel template with the provided arguments
    /// </summary>
    public async Task<KernelExecutionResult> ExecuteTemplateAsync(
        string templateName,
        Accelerator accelerator,
        KernelArguments arguments,
        KernelConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        var compiledKernel = await GetCompiledKernelAsync(templateName, accelerator, cancellationToken);
        if (compiledKernel == null)
        {
            return new KernelExecutionResult(
                Success: false,
                ErrorMessage: $"Failed to compile kernel template: {templateName}");
        }
        
        try
        {
            _logger.LogDebug("Executing kernel template: {Template}", templateName);
            
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            // Execute the kernel
            var stream = accelerator.DefaultStream;
            var kernelConfig = config ?? GetDefaultConfig(accelerator, arguments);
            
            compiledKernel.Kernel(stream, kernelConfig, arguments.ToArray());
            stream.Synchronize();
            
            stopwatch.Stop();
            
            _logger.LogDebug("Kernel template execution completed: {Template} in {Time}ms",
                templateName, stopwatch.ElapsedMilliseconds);
            
            return new KernelExecutionResult(
                Success: true,
                Timing: new KernelTiming(
                    QueueTime: TimeSpan.Zero,
                    KernelTime: stopwatch.Elapsed,
                    TotalTime: stopwatch.Elapsed),
                Metadata: new Dictionary<string, object>
                {
                    ["template"] = templateName,
                    ["accelerator"] = accelerator.AcceleratorType.ToString()
                });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute kernel template: {Template}", templateName);
            return new KernelExecutionResult(
                Success: false,
                ErrorMessage: ex.Message);
        }
    }

    /// <summary>
    /// Gets all available kernel templates
    /// </summary>
    public IReadOnlyDictionary<string, KernelTemplate> GetAvailableTemplates()
    {
        return _templates;
    }

    /// <summary>
    /// Gets templates by category
    /// </summary>
    public IEnumerable<KernelTemplate> GetTemplatesByCategory(KernelCategory category)
    {
        return _templates.Values.Where(t => t.Category == category);
    }

    private KernelCategory GetCategoryFromName(string name)
    {
        if (name.StartsWith("vector/")) return KernelCategory.Vector;
        if (name.StartsWith("reduction/")) return KernelCategory.Reduction;
        if (name.StartsWith("matrix/")) return KernelCategory.Matrix;
        if (name.StartsWith("image/")) return KernelCategory.Image;
        return KernelCategory.Custom;
    }

    private KernelConfig GetDefaultConfig(Accelerator accelerator, KernelArguments arguments)
    {
        // Determine optimal configuration based on accelerator and workload
        var workSize = arguments.GetWorkSize();
        var blockSize = Math.Min(256, accelerator.MaxNumThreadsPerGroup);
        var gridSize = (workSize + blockSize - 1) / blockSize;
        
        return new KernelConfig(gridSize, blockSize);
    }

    public void Dispose()
    {
        _compilationLock?.Dispose();
        _compiledKernels.Clear();
        _templates.Clear();
    }
}

/// <summary>
/// Represents a kernel template
/// </summary>
public class KernelTemplate
{
    public required string Name { get; init; }
    public required Type Type { get; init; }
    public required MethodInfo Method { get; init; }
    public required KernelCategory Category { get; init; }
}

/// <summary>
/// Represents compiled kernel information
/// </summary>
public class CompiledKernelInfo
{
    public required KernelTemplate Template { get; init; }
    public required object Kernel { get; init; }
    public required Accelerator Accelerator { get; init; }
    public required TimeSpan CompilationTime { get; init; }
}

/// <summary>
/// Kernel categories
/// </summary>
public enum KernelCategory
{
    Vector,
    Reduction,
    Matrix,
    Image,
    Custom
}

/// <summary>
/// Kernel arguments wrapper
/// </summary>
public class KernelArguments
{
    private readonly List<object> _arguments = new();
    
    public void Add(object argument) => _arguments.Add(argument);
    
    public object[] ToArray() => _arguments.ToArray();
    
    public int GetWorkSize()
    {
        // Try to determine work size from first array argument
        foreach (var arg in _arguments)
        {
            if (arg is ArrayView<float> view)
                return (int)view.Length;
            if (arg is ArrayView2D<float, Stride2D.DenseX> view2d)
                return view2d.IntExtent.X * view2d.IntExtent.Y;
        }
        return 1024; // Default
    }
}