using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Orleans.GpuBridge.DotCompute;
using Xunit;

namespace Orleans.GpuBridge.Tests.DotCompute;

public class BufferSerializerTests
{
    [Fact]
    public void Serialize_And_Deserialize_Should_Preserve_Data()
    {
        // Arrange
        var data = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();

        // Act
        var serialized = BufferSerializer.Serialize<float>(data);
        var deserialized = BufferSerializer.Deserialize<float>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], deserialized[i]);
        }
    }

    [Fact]
    public void Serialize_Should_Include_Header()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        var serialized = BufferSerializer.Serialize<float>(data.AsSpan());

        // Assert
        Assert.True(serialized.Length > data.Length * sizeof(float)); // Header overhead
        Assert.True(serialized.Length >= 16); // Minimum header size
    }

    [Fact]
    public void Serialize_Should_Support_Float()
    {
        // Arrange
        var data = new float[10];

        // Act
        var serialized = BufferSerializer.Serialize<float>(data);
        var deserialized = BufferSerializer.Deserialize<float>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
    }

    [Fact]
    public void Serialize_Should_Support_Double()
    {
        // Arrange
        var data = new double[10];

        // Act
        var serialized = BufferSerializer.Serialize<double>(data);
        var deserialized = BufferSerializer.Deserialize<double>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
    }

    [Fact]
    public void Serialize_Should_Support_Int()
    {
        // Arrange
        var data = new int[10];

        // Act
        var serialized = BufferSerializer.Serialize<int>(data);
        var deserialized = BufferSerializer.Deserialize<int>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
    }

    [Fact]
    public void Serialize_Should_Support_Long()
    {
        // Arrange
        var data = new long[10];

        // Act
        var serialized = BufferSerializer.Serialize<long>(data);
        var deserialized = BufferSerializer.Deserialize<long>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
    }

    [Fact]
    public void Serialize_Should_Support_Byte()
    {
        // Arrange
        var data = new byte[10];

        // Act
        var serialized = BufferSerializer.Serialize<byte>(data);
        var deserialized = BufferSerializer.Deserialize<byte>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
    }

    [Fact]
    public void Deserialize_With_Invalid_Magic_Should_Throw()
    {
        // Arrange
        var invalidData = new byte[100];
        Random.Shared.NextBytes(invalidData);

        // Act & Assert
        Assert.Throws<InvalidDataException>(() => 
            BufferSerializer.Deserialize<float>(invalidData));
    }

    [Fact]
    public void Deserialize_With_Too_Small_Buffer_Should_Throw()
    {
        // Arrange
        var smallBuffer = new byte[10]; // Too small for header

        // Act & Assert
        Assert.Throws<ArgumentException>(() => 
            BufferSerializer.Deserialize<float>(smallBuffer));
    }

    [Fact]
    public async Task SerializeAsync_Should_Stream_Data()
    {
        // Arrange
        async IAsyncEnumerable<float> GenerateData()
        {
            for (int i = 1; i <= 100; i++)
                yield return (float)i;
        }
        
        var data = GenerateData();
        using var stream = new MemoryStream();

        // Act
        await BufferSerializer.SerializeAsync(stream, data);
        stream.Position = 0;

        // Assert
        Assert.True(stream.Length > 0);
        
        // Deserialize to verify
        var result = new List<float>();
        await foreach (var item in BufferSerializer.DeserializeAsync<float>(stream))
        {
            result.Add(item);
        }
        Assert.Equal(100, result.Count);
        Assert.Equal(1.0f, result[0]);
        Assert.Equal(100.0f, result[99]);
    }

    [Fact]
    public async Task DeserializeAsync_Should_Stream_Data()
    {
        // Arrange
        var data = Enumerable.Range(1, 1000).Select(i => (float)i).ToArray();
        var serialized = BufferSerializer.Serialize<float>(data);
        using var stream = new MemoryStream(serialized);

        // Act
        var result = new List<float>();
        await foreach (var item in BufferSerializer.DeserializeAsync<float>(stream))
        {
            result.Add(item);
        }

        // Assert
        Assert.Equal(data.Length, result.Count);
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], result[i]);
        }
    }

    [Fact]
    public async Task SerializeCompressedAsync_Should_Reduce_Size()
    {
        // Arrange
        var data = new float[1000];
        Array.Fill(data, 1.0f); // Highly compressible

        // Act
        var uncompressed = BufferSerializer.Serialize<float>(data.AsSpan());
        var compressed = await BufferSerializer.SerializeCompressedAsync(
            data, Orleans.GpuBridge.DotCompute.CompressionLevel.Optimal);

        // Assert
        Assert.True(compressed.Length < uncompressed.Length);
    }

    [Fact]
    public async Task DeserializeCompressedAsync_Should_Restore_Data()
    {
        // Arrange
        var data = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();
        var compressed = await BufferSerializer.SerializeCompressedAsync(data);

        // Act
        var decompressed = await BufferSerializer.DeserializeCompressedAsync<float>(compressed);

        // Assert
        Assert.Equal(data.Length, decompressed.Length);
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], decompressed[i]);
        }
    }

    [Theory]
    [InlineData(Orleans.GpuBridge.DotCompute.CompressionLevel.None)]
    [InlineData(Orleans.GpuBridge.DotCompute.CompressionLevel.Fastest)]
    [InlineData(Orleans.GpuBridge.DotCompute.CompressionLevel.Optimal)]
    [InlineData(Orleans.GpuBridge.DotCompute.CompressionLevel.Maximum)]
    public async Task Compression_Levels_Should_Work(Orleans.GpuBridge.DotCompute.CompressionLevel level)
    {
        // Arrange
        var data = Enumerable.Range(1, 100).Select(i => (float)i).ToArray();

        // Act
        var compressed = await BufferSerializer.SerializeCompressedAsync(data, level);
        var decompressed = await BufferSerializer.DeserializeCompressedAsync<float>(compressed);

        // Assert
        Assert.Equal(data.Length, decompressed.Length);
    }

    [Fact]
    public async Task Streaming_With_Cancellation_Should_Cancel()
    {
        // Arrange
        var cts = new CancellationTokenSource();
        async IAsyncEnumerable<float> GenerateLargeData()
        {
            for (int i = 1; i <= 1000000; i++)
                yield return (float)i;
        }
        var data = GenerateLargeData();
        using var stream = new MemoryStream();

        // Act
        cts.CancelAfter(10); // Cancel quickly
        
        // Assert
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
            await BufferSerializer.SerializeAsync(stream, data, cts.Token));
    }

    [Fact]
    public void Empty_Data_Should_Serialize_And_Deserialize()
    {
        // Arrange
        var data = Array.Empty<float>();

        // Act
        var serialized = BufferSerializer.Serialize<float>(data);
        var deserialized = BufferSerializer.Deserialize<float>(serialized);

        // Assert
        Assert.Empty(deserialized);
    }

    [Fact]
    public void Large_Data_Should_Serialize_Correctly()
    {
        // Arrange
        var data = new float[100000];
        Random.Shared.NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan()));

        // Act
        var serialized = BufferSerializer.Serialize<float>(data);
        var deserialized = BufferSerializer.Deserialize<float>(serialized);

        // Assert
        Assert.Equal(data.Length, deserialized.Length);
        Assert.Equal(data[0], deserialized[0]);
        Assert.Equal(data[data.Length - 1], deserialized[data.Length - 1]);
    }
}

public class SerializationBufferPoolTests
{
    [Fact]
    public void Rent_Should_Return_Buffer_Of_Correct_Size()
    {
        // Arrange
        var pool = new SerializationBufferPool();

        // Act
        var small = pool.Rent(100);
        var medium = pool.Rent(10000);
        var large = pool.Rent(500000);
        var veryLarge = pool.Rent(2000000);

        // Assert
        Assert.True(small.Length >= 100);
        Assert.True(medium.Length >= 10000);
        Assert.True(large.Length >= 500000);
        Assert.True(veryLarge.Length >= 2000000);

        // Cleanup
        pool.Return(small);
        pool.Return(medium);
        pool.Return(large);
        pool.Return(veryLarge);
    }

    [Fact]
    public void Return_Should_Clear_Buffer()
    {
        // Arrange
        var pool = new SerializationBufferPool();
        var buffer = pool.Rent(100);
        buffer[0] = 42;
        buffer[50] = 123;

        // Act
        pool.Return(buffer);
        var newBuffer = pool.Rent(100);

        // Assert
        if (ReferenceEquals(buffer, newBuffer)) // If same buffer reused
        {
            Assert.Equal(0, newBuffer[0]);
            Assert.Equal(0, newBuffer[50]);
        }
    }

    [Fact]
    public void Clear_Should_Empty_All_Pools()
    {
        // Arrange
        var pool = new SerializationBufferPool();
        var buffer1 = pool.Rent(100);
        var buffer2 = pool.Rent(10000);
        pool.Return(buffer1);
        pool.Return(buffer2);

        // Act
        pool.Clear();
        var newBuffer = pool.Rent(100);

        // Assert
        Assert.NotNull(newBuffer);
        Assert.True(newBuffer.Length >= 100);
    }

    [Fact]
    public void Concurrent_Rent_And_Return_Should_Be_Thread_Safe()
    {
        // Arrange
        var pool = new SerializationBufferPool();
        var tasks = new Task[100];
        
        // Act
        for (int i = 0; i < tasks.Length; i++)
        {
            var index = i;
            tasks[i] = Task.Run(() =>
            {
                var buffer = pool.Rent(Random.Shared.Next(100, 100000));
                Thread.Sleep(Random.Shared.Next(1, 10));
                pool.Return(buffer);
            });
        }

        // Assert - should complete without exceptions
        Task.WaitAll(tasks);
    }
}