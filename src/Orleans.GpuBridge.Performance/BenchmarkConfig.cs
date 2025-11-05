using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// BenchmarkDotNet configuration for performance tests
/// </summary>
public class BenchmarkConfig : ManualConfig
{
    public BenchmarkConfig()
    {
        AddJob(Job.Default
            .WithGcServer(true)
            .WithGcConcurrent(true)
            .WithGcForce(false));

        AddColumn(
            StatisticColumn.Mean,
            StatisticColumn.Min,
            StatisticColumn.Max,
            StatisticColumn.P95,
            BaselineColumn.RatioMean,
            RankColumn.Arabic);

        AddExporter(MarkdownExporter.GitHub);
        AddExporter(HtmlExporter.Default);
    }
}
