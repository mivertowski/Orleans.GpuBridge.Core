using System;

namespace Orleans.GpuBridge.Backends.DotCompute.Serialization;

internal static class CompressionExtensions
{
    public static System.IO.Compression.CompressionLevel ToCompressionLevel(
        this CompressionLevel level)
    {
        return level switch
        {
            CompressionLevel.None => System.IO.Compression.CompressionLevel.NoCompression,
            CompressionLevel.Fastest => System.IO.Compression.CompressionLevel.Fastest,
            CompressionLevel.Maximum => System.IO.Compression.CompressionLevel.SmallestSize,
            _ => System.IO.Compression.CompressionLevel.Optimal
        };
    }
}
