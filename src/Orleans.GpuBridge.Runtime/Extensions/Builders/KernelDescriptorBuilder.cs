using Orleans.GpuBridge.Abstractions;

namespace Orleans.GpuBridge.Runtime.Builders;

/// <summary>
/// Builder for kernel descriptors
/// </summary>
public class KernelDescriptorBuilder
{
    private readonly KernelDescriptor _descriptor = new();
    
    public KernelDescriptorBuilder Id(string id)
    {
        _descriptor.Id = new KernelId(id);
        return this;
    }
    
    public KernelDescriptorBuilder Input<TIn>() where TIn : notnull
    {
        _descriptor.InType = typeof(TIn);
        return this;
    }
    
    public KernelDescriptorBuilder Output<TOut>() where TOut : notnull
    {
        _descriptor.OutType = typeof(TOut);
        return this;
    }
    
    public KernelDescriptorBuilder WithFactory<TKernel>(Func<IServiceProvider, TKernel> factory)
        where TKernel : class
    {
        _descriptor.Factory = sp => factory(sp);
        return this;
    }
    
    public KernelDescriptorBuilder WithBatchSize(int size)
    {
        // Store batch size in descriptor for optimization hints
        // This would be used by the runtime for performance tuning
        return this;
    }
    
    public KernelDescriptor Build() => _descriptor;
}