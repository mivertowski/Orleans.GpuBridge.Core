using BenchmarkDotNet.Jobs;

namespace Orleans.GpuBridge.Tests.TestingFramework;

/// <summary>
/// Extension methods for BenchmarkDotNet Job configuration
/// </summary>
public static class BenchmarkExtensions
{
    /// <summary>
    /// Configures the job to use server GC
    /// </summary>
    public static Job WithGcServer(this Job job, bool useServerGc)
    {
        return job.WithGcMode(new BenchmarkDotNet.Environments.GcMode 
        { 
            Server = useServerGc 
        });
    }

    /// <summary>
    /// Configures the job to run cleanup after each invocation
    /// </summary>
    public static Job WithCleanup(this Job job, bool cleanup = true)
    {
        return job.WithGcMode(new BenchmarkDotNet.Environments.GcMode 
        { 
            Force = cleanup 
        });
    }

    /// <summary>
    /// Configures the target count for benchmark runs
    /// </summary>
    public static Job WithTargetCount(this Job job, int targetCount)
    {
        return job.WithIterationCount(targetCount);
    }

    /// <summary>
    /// Configures the maximum iteration count
    /// </summary>
    public static Job WithMaxIterationCount(this Job job, int maxCount)
    {
        return job.WithMaxIterationCount(maxCount);
    }

    /// <summary>
    /// Configures the minimum iteration count
    /// </summary>
    public static Job WithMinIterationCount(this Job job, int minCount)
    {
        return job.WithMinIterationCount(minCount);
    }

    /// <summary>
    /// Configures whether to evaluate overhead
    /// </summary>
    public static Job WithEvaluateOverhead(this Job job, bool evaluate = true)
    {
        return job.WithStrategy(BenchmarkDotNet.Engines.RunStrategy.Throughput);
    }
}