namespace Orleans.GpuBridge.Analyzers;

/// <summary>
/// Diagnostic IDs for Orleans.GpuBridge analyzers.
/// </summary>
public static class DiagnosticIds
{
    // Correctness Analyzers (Errors) - OGBA001-OGBA099
    public const string ConfigureAwaitInGrain = "OGBA001";
    public const string QueueCapacityNotPowerOfTwo = "OGBA002";
    public const string MessageSizeNotPowerOfTwo = "OGBA003";
    public const string QueueCapacityOutOfRange = "OGBA004";
    public const string MessageSizeOutOfRange = "OGBA005";
    public const string ThreadsPerActorExceedsLimit = "OGBA006";
    public const string MissingValidateOptionsRegistration = "OGBA007";
    public const string SynchronousBlockingInGrain = "OGBA008";

    // Performance Analyzers (Warnings) - OGBA101-OGBA199
    public const string TemporalOrderingOverhead = "OGBA101";
    public const string QueueCapacityTooLarge = "OGBA102";
    public const string QueueCapacityTooSmall = "OGBA103";
    public const string MultipleSmallMessages = "OGBA104";
    public const string LargeMessageSize = "OGBA105";
    public const string InefficientSerialization = "OGBA106";

    // Best Practice Analyzers (Info) - OGBA201-OGBA299
    public const string ActorNotUsingGpuResidentState = "OGBA201";
    public const string MissingTelemetryRegistration = "OGBA202";
    public const string MissingHealthCheckRegistration = "OGBA203";
    public const string ActorNotImplementingDisposal = "OGBA204";
    public const string HlcTimestampNotUsed = "OGBA205";
    public const string GpuMemoryAllocationWithoutPooling = "OGBA206";
}

/// <summary>
/// Category names for diagnostics.
/// </summary>
public static class DiagnosticCategories
{
    public const string Correctness = "Orleans.GpuBridge.Correctness";
    public const string Performance = "Orleans.GpuBridge.Performance";
    public const string BestPractices = "Orleans.GpuBridge.BestPractices";
}
