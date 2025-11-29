namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Compute kernel interface
/// </summary>
public interface IComputeKernel : IDisposable
{
    string Name { get; }

    void SetArgument(int index, IComputeBuffer<float> buffer);
    void SetArgument(int index, IComputeBuffer<double> buffer);
    void SetArgument(int index, IComputeBuffer<int> buffer);
    void SetArgument(int index, float value);
    void SetArgument(int index, double value);
    void SetArgument(int index, int value);
}