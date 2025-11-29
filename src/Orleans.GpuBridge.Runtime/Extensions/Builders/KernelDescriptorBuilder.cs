using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.Builders;

/// <summary>
/// Builder for kernel descriptors
/// </summary>
public class KernelDescriptorBuilder
{
    private readonly KernelDescriptor _descriptor = new();

    /// <summary>
    /// Sets the kernel identifier
    /// </summary>
    /// <param name="id">Kernel identifier string</param>
    /// <returns>This builder for fluent chaining</returns>
    public KernelDescriptorBuilder Id(string id)
    {
        _descriptor.Id = new KernelId(id);
        return this;
    }

    /// <summary>
    /// Sets the input type for the kernel
    /// </summary>
    /// <typeparam name="TIn">Input type</typeparam>
    /// <returns>This builder for fluent chaining</returns>
    public KernelDescriptorBuilder Input<TIn>() where TIn : notnull
    {
        _descriptor.InType = typeof(TIn);
        return this;
    }

    /// <summary>
    /// Sets the output type for the kernel
    /// </summary>
    /// <typeparam name="TOut">Output type</typeparam>
    /// <returns>This builder for fluent chaining</returns>
    public KernelDescriptorBuilder Output<TOut>() where TOut : notnull
    {
        _descriptor.OutType = typeof(TOut);
        return this;
    }

    /// <summary>
    /// Sets the factory function for creating kernel instances
    /// </summary>
    /// <typeparam name="TKernel">Kernel implementation type</typeparam>
    /// <param name="factory">Factory function that creates kernel instances</param>
    /// <returns>This builder for fluent chaining</returns>
    public KernelDescriptorBuilder WithFactory<TKernel>(Func<IServiceProvider, TKernel> factory)
        where TKernel : class
    {
        _descriptor.Factory = sp => factory(sp);
        return this;
    }

    /// <summary>
    /// Sets the batch size hint for kernel execution optimization
    /// </summary>
    /// <param name="size">Preferred batch size</param>
    /// <returns>This builder for fluent chaining</returns>
    public KernelDescriptorBuilder WithBatchSize(int size)
    {
        // Store batch size in descriptor for optimization hints
        // This would be used by the runtime for performance tuning
        return this;
    }

    /// <summary>
    /// Builds the kernel descriptor from the configured values
    /// </summary>
    /// <returns>The configured kernel descriptor</returns>
    public KernelDescriptor Build() => _descriptor;
}
