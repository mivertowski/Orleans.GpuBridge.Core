namespace Orleans.GpuBridge.Backends.DotCompute.Enums;

/// <summary>
/// Enumeration of supported vector operations
/// </summary>
public enum VectorOperation
{
    /// <summary>
    /// Element-wise addition
    /// </summary>
    Add,
    
    /// <summary>
    /// Element-wise subtraction
    /// </summary>
    Subtract,
    
    /// <summary>
    /// Element-wise multiplication
    /// </summary>
    Multiply,
    
    /// <summary>
    /// Element-wise division
    /// </summary>
    Divide,
    
    /// <summary>
    /// Fused multiply-add operation
    /// </summary>
    FusedMultiplyAdd,
    
    /// <summary>
    /// Square root
    /// </summary>
    Sqrt,
    
    /// <summary>
    /// Absolute value
    /// </summary>
    Abs,
    
    /// <summary>
    /// Power operation
    /// </summary>
    Power,
    
    /// <summary>
    /// Exponential
    /// </summary>
    Exp,
    
    /// <summary>
    /// Natural logarithm
    /// </summary>
    Log,
    
    /// <summary>
    /// Sine function
    /// </summary>
    Sin,
    
    /// <summary>
    /// Cosine function
    /// </summary>
    Cos,
    
    /// <summary>
    /// Dot product
    /// </summary>
    DotProduct,
    
    /// <summary>
    /// Vector norm (L2)
    /// </summary>
    Norm,
    
    /// <summary>
    /// Clamp values to range
    /// </summary>
    Clamp
}