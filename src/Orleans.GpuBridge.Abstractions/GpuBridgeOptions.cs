namespace Orleans.GpuBridge.Abstractions;
public sealed class GpuBridgeOptions { public bool PreferGpu { get; set; } = true; public int DefaultMicroBatch { get; set; } = 8192; }
