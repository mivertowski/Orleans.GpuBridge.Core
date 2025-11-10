namespace Orleans.GpuBridge.Tests.RC2.Infrastructure;

/// <summary>
/// xUnit collection definition for sharing Orleans test cluster across test classes.
/// This ensures a single cluster instance is used for all grain tests, improving performance.
/// </summary>
[CollectionDefinition("ClusterCollection")]
public sealed class ClusterCollection : ICollectionFixture<ClusterFixture>
{
    // This class has no code, and is never created. Its purpose is simply
    // to be the place to apply [CollectionDefinition] and all the
    // ICollectionFixture<> interfaces.
}
