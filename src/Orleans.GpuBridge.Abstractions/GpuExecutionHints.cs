namespace Orleans.GpuBridge.Abstractions;
public sealed record GpuExecutionHints(int? PreferredDevice=null, bool HighPriority=false, int? MaxMicroBatch=null, bool Persistent=true);
