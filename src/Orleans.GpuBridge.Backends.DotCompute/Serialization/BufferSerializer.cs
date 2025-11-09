using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Orleans.GpuBridge.Backends.DotCompute.Serialization;

/// <summary>
/// High-performance binary serializer for GPU buffers
/// </summary>
public static class BufferSerializer
{
    private const int HeaderSize = 16;
    private const uint MagicNumber = 0x47505542; // "GPUB"
    private const byte CurrentVersion = 1;

    /// <summary>
    /// Serializes data to a binary format optimized for GPU transfer
    /// </summary>
    public static byte[] Serialize<T>(ReadOnlySpan<T> data) where T : unmanaged
    {
        var elementSize = Unsafe.SizeOf<T>();
        var dataSize = data.Length * elementSize;
        var totalSize = HeaderSize + dataSize;

        var buffer = new byte[totalSize];
        var span = buffer.AsSpan();

        // Write header
        WriteHeader(span, data.Length, elementSize, typeof(T));

        // Write data
        unsafe
        {
            fixed (T* src = data)
            fixed (byte* dst = &buffer[HeaderSize])
            {
                Buffer.MemoryCopy(src, dst, dataSize, dataSize);
            }
        }

        return buffer;
    }

    /// <summary>
    /// Deserializes data from binary format
    /// </summary>
    public static T[] Deserialize<T>(ReadOnlySpan<byte> buffer) where T : unmanaged
    {
        if (buffer.Length < HeaderSize)
            throw new ArgumentException("Buffer too small for header");

        // Read and validate header
        var header = ReadHeader(buffer);

        if (header.Magic != MagicNumber)
            throw new InvalidDataException("Invalid magic number");

        if (header.Version != CurrentVersion)
            throw new InvalidDataException($"Unsupported version: {header.Version}");

        var expectedElementSize = Unsafe.SizeOf<T>();
        if (header.ElementSize != expectedElementSize)
            throw new InvalidDataException(
                $"Element size mismatch. Expected {expectedElementSize}, got {header.ElementSize}");

        // Read data
        var result = new T[header.ElementCount];
        var dataSpan = buffer.Slice(HeaderSize);

        unsafe
        {
            fixed (byte* src = dataSpan)
            fixed (T* dst = result)
            {
                var size = header.ElementCount * header.ElementSize;
                Buffer.MemoryCopy(src, dst, size, size);
            }
        }

        return result;
    }

    /// <summary>
    /// Async serialization with streaming support
    /// </summary>
    public static async Task SerializeAsync<T>(
        Stream stream,
        IAsyncEnumerable<T> data,
        CancellationToken ct = default) where T : unmanaged
    {
        var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var elementSize = Unsafe.SizeOf<T>();
        var buffer = ArrayPool<T>.Shared.Rent(4096);
        var byteBuffer = ArrayPool<byte>.Shared.Rent(4096 * elementSize);

        try
        {
            // Write placeholder header
            writer.Write(MagicNumber);
            writer.Write(CurrentVersion);
            writer.Write(0); // Element count (will update)
            writer.Write(elementSize);
            writer.Write(GetTypeId<T>());

            var count = 0;
            var bufferIndex = 0;

            await foreach (var item in data.WithCancellation(ct))
            {
                buffer[bufferIndex++] = item;

                if (bufferIndex >= buffer.Length)
                {
                    // Flush buffer
                    await WriteBufferAsync(stream, buffer, bufferIndex, byteBuffer, ct);
                    count += bufferIndex;
                    bufferIndex = 0;
                }
            }

            // Flush remaining
            if (bufferIndex > 0)
            {
                await WriteBufferAsync(stream, buffer, bufferIndex, byteBuffer, ct);
                count += bufferIndex;
            }

            // Update count in header
            stream.Position = 5;
            writer.Write(count);
            stream.Position = stream.Length;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(buffer);
            ArrayPool<byte>.Shared.Return(byteBuffer);
            writer.Dispose();
        }
    }

    /// <summary>
    /// Async deserialization with streaming support
    /// </summary>
    public static async IAsyncEnumerable<T> DeserializeAsync<T>(
        Stream stream,
        [EnumeratorCancellation] CancellationToken ct = default) where T : unmanaged
    {
        var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read header
        var magic = reader.ReadUInt32();
        if (magic != MagicNumber)
            throw new InvalidDataException("Invalid magic number");

        var version = reader.ReadByte();
        if (version != CurrentVersion)
            throw new InvalidDataException($"Unsupported version: {version}");

        var count = reader.ReadInt32();
        var elementSize = reader.ReadInt32();
        var typeId = reader.ReadInt32();

        var expectedElementSize = Unsafe.SizeOf<T>();
        if (elementSize != expectedElementSize)
            throw new InvalidDataException(
                $"Element size mismatch. Expected {expectedElementSize}, got {elementSize}");

        // Read data in chunks
        var chunkSize = Math.Min(4096, count);
        var buffer = ArrayPool<byte>.Shared.Rent(chunkSize * elementSize);
        var items = new T[chunkSize];

        try
        {
            var remaining = count;
            while (remaining > 0 && !ct.IsCancellationRequested)
            {
                var toRead = Math.Min(chunkSize, remaining);
                var bytesToRead = toRead * elementSize;

                var bytesRead = await stream.ReadAsync(
                    buffer.AsMemory(0, bytesToRead), ct);

                if (bytesRead < bytesToRead)
                    throw new EndOfStreamException();

                // Convert bytes to items
                unsafe
                {
                    fixed (byte* src = buffer)
                    fixed (T* dst = items)
                    {
                        Buffer.MemoryCopy(src, dst, bytesToRead, bytesToRead);
                    }
                }

                for (int i = 0; i < toRead; i++)
                {
                    yield return items[i];
                }

                remaining -= toRead;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
            reader.Dispose();
        }
    }

    /// <summary>
    /// Compressed serialization for network transfer
    /// </summary>
    public static async Task<byte[]> SerializeCompressedAsync<T>(
        T[] data,
        CompressionLevel level = CompressionLevel.Optimal,
        CancellationToken ct = default) where T : unmanaged
    {
        using var ms = new MemoryStream();

        // Write uncompressed header
        await ms.WriteAsync(BitConverter.GetBytes(MagicNumber), ct);
        await ms.WriteAsync(new[] { CurrentVersion }, ct);
        await ms.WriteAsync(BitConverter.GetBytes(data.Length), ct);
        await ms.WriteAsync(BitConverter.GetBytes(Unsafe.SizeOf<T>()), ct);

        // Compress data
        using (var compressor = new System.IO.Compression.BrotliStream(
            ms, level.ToCompressionLevel(), leaveOpen: true))
        {
            var bytes = MemoryMarshal.AsBytes(data.AsSpan()).ToArray();
            await compressor.WriteAsync(bytes, ct);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Decompressed deserialization
    /// </summary>
    public static async Task<T[]> DeserializeCompressedAsync<T>(
        byte[] compressedData,
        CancellationToken ct = default) where T : unmanaged
    {
        using var ms = new MemoryStream(compressedData);

        // Read header
        var headerBuffer = new byte[HeaderSize];
        await ms.ReadAsync(headerBuffer, ct);

        var magic = BitConverter.ToUInt32(headerBuffer, 0);
        if (magic != MagicNumber)
            throw new InvalidDataException("Invalid magic number");

        var version = headerBuffer[4];
        if (version != CurrentVersion)
            throw new InvalidDataException($"Unsupported version: {version}");

        var count = BitConverter.ToInt32(headerBuffer, 5);
        var elementSize = BitConverter.ToInt32(headerBuffer, 9);

        // Decompress data
        using var decompressor = new System.IO.Compression.BrotliStream(
            ms, System.IO.Compression.CompressionMode.Decompress);

        var dataSize = count * elementSize;
        var buffer = new byte[dataSize];

        var totalRead = 0;
        while (totalRead < dataSize)
        {
            var read = await decompressor.ReadAsync(
                buffer.AsMemory(totalRead, dataSize - totalRead), ct);

            if (read == 0)
                throw new EndOfStreamException();

            totalRead += read;
        }

        // Convert to array
        var result = new T[count];
        unsafe
        {
            fixed (byte* src = buffer)
            fixed (T* dst = result)
            {
                Buffer.MemoryCopy(src, dst, dataSize, dataSize);
            }
        }

        return result;
    }

    private static void WriteHeader(Span<byte> buffer, int count, int elementSize, Type type)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(buffer, MagicNumber);
        buffer[4] = CurrentVersion;
        BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(5), count);
        BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(9), elementSize);
        BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(13), GetTypeId(type));
    }

    private static Header ReadHeader(ReadOnlySpan<byte> buffer)
    {
        return new Header
        {
            Magic = BinaryPrimitives.ReadUInt32LittleEndian(buffer),
            Version = buffer[4],
            ElementCount = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(5)),
            ElementSize = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(9)),
            TypeId = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(13))
        };
    }

    private static async Task WriteBufferAsync<T>(
        Stream stream,
        T[] buffer,
        int count,
        byte[] byteBuffer,
        CancellationToken ct) where T : unmanaged
    {
        var bytes = count * Unsafe.SizeOf<T>();

        unsafe
        {
            fixed (T* src = buffer)
            fixed (byte* dst = byteBuffer)
            {
                Buffer.MemoryCopy(src, dst, bytes, bytes);
            }
        }

        await stream.WriteAsync(byteBuffer.AsMemory(0, bytes), ct);
    }

    private static int GetTypeId<T>() => GetTypeId(typeof(T));

    private static int GetTypeId(Type type)
    {
        return type.Name switch
        {
            "Byte" => 1,
            "SByte" => 2,
            "Int16" => 3,
            "UInt16" => 4,
            "Int32" => 5,
            "UInt32" => 6,
            "Int64" => 7,
            "UInt64" => 8,
            "Single" => 9,
            "Double" => 10,
            "Decimal" => 11,
            "Boolean" => 12,
            _ => type.GetHashCode()
        };
    }

    private readonly struct Header
    {
        public uint Magic { get; init; }
        public byte Version { get; init; }
        public int ElementCount { get; init; }
        public int ElementSize { get; init; }
        public int TypeId { get; init; }
    }
}
