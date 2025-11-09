namespace Orleans.GpuBridge.Backends.DotCompute.Execution;

/// <summary>
/// Work dimensions for DotCompute kernel execution
/// </summary>
/// <param name="GlobalSize">Global work size dimensions</param>
/// <param name="LocalSize">Local work size dimensions (optional)</param>
internal record WorkDimensions(int[] GlobalSize, int[]? LocalSize);
