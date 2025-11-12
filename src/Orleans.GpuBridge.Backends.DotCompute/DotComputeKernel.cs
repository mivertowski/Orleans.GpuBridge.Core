using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using DotCompute.Abstractions;
using DotCompute.Abstractions.Kernels;
using DotCompute.Abstractions.Kernels.Types;
using DotCompute.Abstractions.Memory;
using Orleans.GpuBridge.Abstractions.Kernels;

namespace Orleans.GpuBridge.Backends.DotCompute;

/// <summary>
/// DotCompute kernel adapter that wraps IAccelerator and ICompiledKernel.
/// Provides Orleans.GpuBridge integration for DotCompute kernels.
/// </summary>
/// <typeparam name="TIn">Input type for kernel execution</typeparam>
/// <typeparam name="TOut">Output type for kernel execution</typeparam>
public sealed class DotComputeKernel<TIn, TOut> : GpuKernelBase<TIn, TOut>
{
    private readonly IAccelerator _accelerator;
    private readonly KernelDefinition _kernelDefinition;
    private readonly CompilationOptions _compilationOptions;
    private readonly Func<TIn, KernelArgument[]> _inputConverter;
    private readonly Func<IUnifiedMemoryBuffer, TOut> _outputConverter;
    private ICompiledKernel? _compiledKernel;
    private bool _isWarmupComplete;

    /// <summary>
    /// Unique kernel identifier
    /// </summary>
    public override string KernelId => _kernelDefinition.Name;

    /// <summary>
    /// Backend provider name
    /// </summary>
    public override string BackendProvider => $"DotCompute/{_accelerator.Info.Name}";

    /// <summary>
    /// Whether kernel uses GPU acceleration (true for non-CPU accelerators)
    /// </summary>
    public override bool IsGpuAccelerated =>
        _accelerator.Type != AcceleratorType.CPU;

    /// <summary>
    /// Creates a new DotCompute kernel adapter.
    /// </summary>
    /// <param name="accelerator">DotCompute accelerator instance</param>
    /// <param name="kernelDefinition">Kernel definition with source code</param>
    /// <param name="compilationOptions">Compilation options (null = default)</param>
    /// <param name="inputConverter">Function to convert TIn to KernelArgument[]</param>
    /// <param name="outputConverter">Function to convert GPU buffer to TOut</param>
    public DotComputeKernel(
        IAccelerator accelerator,
        KernelDefinition kernelDefinition,
        CompilationOptions? compilationOptions = null,
        Func<TIn, KernelArgument[]>? inputConverter = null,
        Func<IUnifiedMemoryBuffer, TOut>? outputConverter = null)
    {
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _kernelDefinition = kernelDefinition ?? throw new ArgumentNullException(nameof(kernelDefinition));
        _compilationOptions = compilationOptions ?? CompilationOptions.Default;

        // Default converters handle simple cases - override for complex scenarios
        _inputConverter = inputConverter ?? DefaultInputConverter;
        _outputConverter = outputConverter ?? DefaultOutputConverter;
    }

    /// <summary>
    /// Initialize kernel by compiling to GPU code.
    /// </summary>
    public override async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();

        if (IsInitialized)
            return;

        try
        {
            // Compile kernel via DotCompute
            _compiledKernel = await _accelerator.CompileKernelAsync(
                _kernelDefinition,
                _compilationOptions,
                cancellationToken);

            await base.InitializeAsync(cancellationToken);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to compile DotCompute kernel '{KernelId}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Execute kernel with single input.
    /// </summary>
    public override async Task<TOut> ExecuteAsync(TIn input, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        if (_compiledKernel == null)
            throw new InvalidOperationException("Kernel not compiled. Call InitializeAsync() first.");

        // Validate input
        var validationResult = ValidateInput(input);
        if (!validationResult.IsValid)
            throw new ArgumentException(validationResult.ErrorMessage, nameof(input));

        IUnifiedMemoryBuffer? outputBuffer = null;

        try
        {
            // Convert input to kernel arguments
            var argumentArray = _inputConverter(input);

            // Create KernelArguments instance with the array
            var kernelArgs = new KernelArguments(argumentArray);

            // Execute kernel
            await _compiledKernel.ExecuteAsync(kernelArgs, cancellationToken);

            // Synchronize to ensure completion
            await _accelerator.SynchronizeAsync(cancellationToken);

            // Find output buffer in arguments (convention: last argument is output)
            var outputArg = argumentArray[^1];
            if (outputArg.Value == null)
                throw new InvalidOperationException("Output argument does not contain a value.");

            // Convert output buffer to TOut
            // The MemoryBuffer property should be accessed if Value is IUnifiedMemoryBuffer
            if (outputArg.Value is IUnifiedMemoryBuffer memBuffer)
            {
                return _outputConverter(memBuffer);
            }

            throw new InvalidOperationException($"Output argument Value is not IUnifiedMemoryBuffer, got: {outputArg.Value.GetType()}");
        }
        finally
        {
            // Cleanup GPU memory
            if (outputBuffer != null)
            {
                _accelerator.Memory.Free(outputBuffer);
            }
        }
    }

    /// <summary>
    /// Execute kernel with batch input (optimized for GPU).
    /// </summary>
    public override async Task<TOut[]> ExecuteBatchAsync(TIn[] inputs, CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        if (_compiledKernel == null)
            throw new InvalidOperationException("Kernel not compiled. Call InitializeAsync() first.");

        // For batch execution, we can optimize by:
        // 1. Allocating GPU memory once for all inputs
        // 2. Batching transfers to reduce PCIe overhead
        // 3. Launching kernel with larger work size

        var results = new TOut[inputs.Length];

        // For now, use simple sequential execution
        // TODO: Implement true batch optimization with single GPU allocation
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = await ExecuteAsync(inputs[i], cancellationToken);
        }

        return results;
    }

    /// <summary>
    /// Warmup kernel for optimal performance (JIT compilation, cache warming).
    /// </summary>
    public override async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        EnsureNotDisposed();

        if (_isWarmupComplete)
            return;

        // DotCompute may perform JIT compilation on first execution
        // Run a dummy execution to warm up GPU caches
        try
        {
            // Create dummy input based on TIn type
            var dummyInput = CreateDummyInput();
            if (dummyInput != null)
            {
                _ = await ExecuteAsync(dummyInput, cancellationToken);
            }

            _isWarmupComplete = true;
        }
        catch
        {
            // Warmup is best-effort - don't fail initialization if it fails
        }
    }

    /// <summary>
    /// Get estimated execution time for input size.
    /// Uses GPU timing provider if available for accurate profiling.
    /// </summary>
    public override long GetEstimatedExecutionTimeMicroseconds(int inputSize)
    {
        // Query GPU timing provider for accurate estimates
        var timingProvider = _accelerator.GetTimingProvider();
        if (timingProvider != null)
        {
            // TODO: Implement profiling via ITimingProvider
            // For now, use heuristic based on accelerator type
        }

        // Heuristic: GPU kernels ~0.1μs per element, CPU ~1μs per element
        return IsGpuAccelerated ? inputSize / 10 : inputSize;
    }

    /// <summary>
    /// Get memory requirements for kernel execution.
    /// </summary>
    public override KernelMemoryRequirements GetMemoryRequirements()
    {
        // Query DotCompute memory manager for accurate estimates
        var memStats = _accelerator.Memory.Statistics;

        // Estimate based on TIn/TOut types
        var inputSize = Marshal.SizeOf<TIn>();
        var outputSize = Marshal.SizeOf<TOut>();

        return new KernelMemoryRequirements(
            InputMemoryBytes: inputSize,
            OutputMemoryBytes: outputSize,
            WorkingMemoryBytes: Math.Max(inputSize, outputSize), // Temporary buffers
            TotalMemoryBytes: inputSize + outputSize + Math.Max(inputSize, outputSize));
    }

    /// <summary>
    /// Dispose kernel resources and free GPU memory.
    /// </summary>
    public override void Dispose()
    {
        if (_compiledKernel != null)
        {
            _compiledKernel.Dispose();
            _compiledKernel = null;
        }

        base.Dispose();
    }

    #region Default Converters

    /// <summary>
    /// Default input converter for simple types.
    /// Override via constructor for complex input types.
    /// </summary>
    private KernelArgument[] DefaultInputConverter(TIn input)
    {
        // For simple types (float[], int[], etc.), create memory buffer and copy data
        if (input is Array array)
        {
            var elementType = array.GetType().GetElementType()!;

            // Allocate GPU memory
            // TODO: Implement proper type conversion and memory allocation

            return new[]
            {
                new KernelArgument
                {
                    Name = "input",
                    Value = input,
                    Type = typeof(TIn),
                    IsDeviceMemory = false
                },
                new KernelArgument
                {
                    Name = "output",
                    Value = null!,
                    Type = typeof(TOut),
                    IsDeviceMemory = true
                }
            };
        }

        throw new NotSupportedException(
            $"Default input converter does not support type {typeof(TIn).Name}. " +
            "Provide custom converter via constructor.");
    }

    /// <summary>
    /// Default output converter for simple types.
    /// Override via constructor for complex output types.
    /// </summary>
    private TOut DefaultOutputConverter(IUnifiedMemoryBuffer buffer)
    {
        // For simple types, read from GPU buffer
        // TODO: Implement proper type conversion and memory reading

        throw new NotSupportedException(
            $"Default output converter does not support type {typeof(TOut).Name}. " +
            "Provide custom converter via constructor.");
    }

    /// <summary>
    /// Create dummy input for warmup based on TIn type.
    /// </summary>
    private TIn? CreateDummyInput()
    {
        // For array types, create small dummy array
        if (typeof(TIn).IsArray)
        {
            var elementType = typeof(TIn).GetElementType()!;
            var dummyArray = Array.CreateInstance(elementType, 16); // Small warmup size
            return (TIn)(object)dummyArray;
        }

        // For value types, return default
        if (typeof(TIn).IsValueType)
        {
            return default;
        }

        return default;
    }

    #endregion
}
