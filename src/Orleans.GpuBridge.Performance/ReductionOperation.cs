namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Specifies the type of reduction operation to perform on vectorized data
/// </summary>
public enum ReductionOperation
{
    /// <summary>
    /// Sum all elements in the vector
    /// </summary>
    Sum,

    /// <summary>
    /// Find the maximum element in the vector
    /// </summary>
    Max,

    /// <summary>
    /// Find the minimum element in the vector
    /// </summary>
    Min
}
