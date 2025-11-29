namespace Orleans.GpuBridge.Runtime.BackendProviders.Interfaces;

/// <summary>
/// Compute kernel interface for GPU kernel execution.
/// </summary>
public interface IComputeKernel : IDisposable
{
    /// <summary>
    /// Gets the name of the kernel entry point.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Sets a float buffer argument at the specified index.
    /// </summary>
    /// <param name="index">The argument index.</param>
    /// <param name="buffer">The float buffer to bind.</param>
    void SetArgument(int index, IComputeBuffer<float> buffer);

    /// <summary>
    /// Sets a double buffer argument at the specified index.
    /// </summary>
    /// <param name="index">The argument index.</param>
    /// <param name="buffer">The double buffer to bind.</param>
    void SetArgument(int index, IComputeBuffer<double> buffer);

    /// <summary>
    /// Sets an integer buffer argument at the specified index.
    /// </summary>
    /// <param name="index">The argument index.</param>
    /// <param name="buffer">The integer buffer to bind.</param>
    void SetArgument(int index, IComputeBuffer<int> buffer);

    /// <summary>
    /// Sets a scalar float argument at the specified index.
    /// </summary>
    /// <param name="index">The argument index.</param>
    /// <param name="value">The float value to set.</param>
    void SetArgument(int index, float value);

    /// <summary>
    /// Sets a scalar double argument at the specified index.
    /// </summary>
    /// <param name="index">The argument index.</param>
    /// <param name="value">The double value to set.</param>
    void SetArgument(int index, double value);

    /// <summary>
    /// Sets a scalar integer argument at the specified index.
    /// </summary>
    /// <param name="index">The argument index.</param>
    /// <param name="value">The integer value to set.</param>
    void SetArgument(int index, int value);
}
