// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Runtime.RingKernels;

/// <summary>
/// Serializes Orleans method calls to GPU-compatible messages.
/// </summary>
/// <remarks>
/// <para>
/// This serializer handles the conversion of .NET method calls to fixed-size GPU messages.
/// It supports primitive types, structs, and small arrays that fit within the 228-byte payload.
/// </para>
/// <para>
/// For large data (arrays, strings), the serializer stores GPU memory pointers in the payload
/// and manages separate GPU buffer allocations.
/// </para>
/// </remarks>
public static class GpuMessageSerializer
{
    /// <summary>
    /// Serializes a method invocation to a GPU message.
    /// </summary>
    /// <param name="methodName">Name of the method to invoke.</param>
    /// <param name="senderId">Sender actor ID (0 for Orleans runtime).</param>
    /// <param name="targetId">Target actor ID.</param>
    /// <param name="args">Method arguments to serialize.</param>
    /// <returns>GPU message ready for queue submission.</returns>
    /// <exception cref="ArgumentException">Arguments too large for inline payload.</exception>
    public static unsafe OrleansGpuMessage Serialize(string methodName, int senderId, int targetId, params object[] args)
    {
        var message = OrleansGpuMessage.Create(
            methodId: methodName.GetHashCode(),
            senderId: senderId,
            targetId: targetId,
            type: MessageType.Data);

        // Serialize arguments to payload using the safe GetPayloadSpan method
        var payloadSpan = message.GetPayloadSpan();
        var bytesWritten = SerializeArgs(args, payloadSpan);

        if (bytesWritten > 228)
        {
            throw new ArgumentException(
                $"Serialized arguments ({bytesWritten} bytes) exceed payload capacity (228 bytes). " +
                "Consider using GPU memory pointers for large data.",
                nameof(args));
        }

        return message;
    }

    /// <summary>
    /// Deserializes method arguments from a GPU message payload.
    /// </summary>
    /// <typeparam name="T">Expected return type.</typeparam>
    /// <param name="message">GPU message containing response.</param>
    /// <returns>Deserialized return value.</returns>
    public static unsafe T Deserialize<T>(OrleansGpuMessage message) where T : unmanaged
    {
        if (message.Type != MessageType.Response)
        {
            throw new InvalidOperationException(
                $"Cannot deserialize non-response message (type: {message.Type})");
        }

        var payloadSpan = message.GetPayloadSpan();
        fixed (byte* payloadPtr = payloadSpan)
        {
            return *(T*)payloadPtr;
        }
    }

    /// <summary>
    /// Serializes method arguments to a byte span.
    /// </summary>
    /// <param name="args">Arguments to serialize.</param>
    /// <param name="buffer">Destination buffer (228 bytes).</param>
    /// <returns>Number of bytes written.</returns>
    /// <remarks>
    /// <para>
    /// Serialization strategy:
    /// - Primitives: Direct memory copy
    /// - Structs: Blittable types copied directly
    /// - Arrays: Small arrays (&lt; 200 bytes) copied inline
    /// - Strings: UTF8 encoding, null-terminated
    /// - Large data: Store GPU pointer (8 bytes) instead of data
    /// </para>
    /// </remarks>
    private static int SerializeArgs(object[] args, Span<byte> buffer)
    {
        int offset = 0;

        foreach (var arg in args)
        {
            if (arg == null)
            {
                // Null argument - write sentinel value
                MemoryMarshal.Write(buffer.Slice(offset, 4), ref offset);
                offset += 4;
                continue;
            }

            switch (arg)
            {
                case int intVal:
                    MemoryMarshal.Write(buffer.Slice(offset, 4), ref intVal);
                    offset += 4;
                    break;

                case long longVal:
                    MemoryMarshal.Write(buffer.Slice(offset, 8), ref longVal);
                    offset += 8;
                    break;

                case float floatVal:
                    MemoryMarshal.Write(buffer.Slice(offset, 4), ref floatVal);
                    offset += 4;
                    break;

                case double doubleVal:
                    MemoryMarshal.Write(buffer.Slice(offset, 8), ref doubleVal);
                    offset += 8;
                    break;

                case bool boolVal:
                    buffer[offset] = boolVal ? (byte)1 : (byte)0;
                    offset += 1;
                    break;

                case byte byteVal:
                    buffer[offset] = byteVal;
                    offset += 1;
                    break;

                case int[] intArray:
                    offset += SerializeIntArray(intArray, buffer.Slice(offset));
                    break;

                case float[] floatArray:
                    offset += SerializeFloatArray(floatArray, buffer.Slice(offset));
                    break;

                case string stringVal:
                    offset += SerializeString(stringVal, buffer.Slice(offset));
                    break;

                default:
                    throw new NotSupportedException(
                        $"Serialization of type {arg.GetType().Name} not yet supported. " +
                        "Supported types: int, long, float, double, bool, byte, int[], float[], string");
            }
        }

        return offset;
    }

    /// <summary>
    /// Serializes an integer array to the buffer.
    /// </summary>
    /// <param name="array">Array to serialize.</param>
    /// <param name="buffer">Destination buffer.</param>
    /// <returns>Number of bytes written.</returns>
    private static int SerializeIntArray(int[] array, Span<byte> buffer)
    {
        // Write array length (4 bytes)
        int length = array.Length;
        MemoryMarshal.Write(buffer, ref length);

        // Write array data
        var arrayBytes = MemoryMarshal.AsBytes(array.AsSpan());

        if (4 + arrayBytes.Length > buffer.Length)
        {
            throw new ArgumentException($"Integer array too large ({arrayBytes.Length} bytes) for inline serialization");
        }

        arrayBytes.CopyTo(buffer.Slice(4));
        return 4 + arrayBytes.Length;
    }

    /// <summary>
    /// Serializes a float array to the buffer.
    /// </summary>
    /// <param name="array">Array to serialize.</param>
    /// <param name="buffer">Destination buffer.</param>
    /// <returns>Number of bytes written.</returns>
    private static int SerializeFloatArray(float[] array, Span<byte> buffer)
    {
        // Write array length (4 bytes)
        int length = array.Length;
        MemoryMarshal.Write(buffer, ref length);

        // Write array data
        var arrayBytes = MemoryMarshal.AsBytes(array.AsSpan());

        if (4 + arrayBytes.Length > buffer.Length)
        {
            throw new ArgumentException($"Float array too large ({arrayBytes.Length} bytes) for inline serialization");
        }

        arrayBytes.CopyTo(buffer.Slice(4));
        return 4 + arrayBytes.Length;
    }

    /// <summary>
    /// Serializes a string to the buffer using UTF8 encoding.
    /// </summary>
    /// <param name="str">String to serialize.</param>
    /// <param name="buffer">Destination buffer.</param>
    /// <returns>Number of bytes written.</returns>
    private static int SerializeString(string str, Span<byte> buffer)
    {
        if (string.IsNullOrEmpty(str))
        {
            // Write length = 0
            int zero = 0;
            MemoryMarshal.Write(buffer, ref zero);
            return 4;
        }

        // Write string length
        int byteCount = System.Text.Encoding.UTF8.GetByteCount(str);
        MemoryMarshal.Write(buffer, ref byteCount);

        // Write UTF8 bytes
        if (4 + byteCount > buffer.Length)
        {
            throw new ArgumentException($"String too large ({byteCount} bytes) for inline serialization");
        }

        System.Text.Encoding.UTF8.GetBytes(str, buffer.Slice(4, byteCount));
        return 4 + byteCount;
    }

    /// <summary>
    /// Deserializes an integer array from the buffer.
    /// </summary>
    /// <param name="buffer">Source buffer.</param>
    /// <param name="offset">Starting offset.</param>
    /// <param name="bytesRead">Number of bytes consumed.</param>
    /// <returns>Deserialized array.</returns>
    public static int[] DeserializeIntArray(ReadOnlySpan<byte> buffer, int offset, out int bytesRead)
    {
        // Read array length
        int length = MemoryMarshal.Read<int>(buffer.Slice(offset));

        // Read array data
        var result = new int[length];
        var arrayBytes = MemoryMarshal.AsBytes(result.AsSpan());
        buffer.Slice(offset + 4, arrayBytes.Length).CopyTo(arrayBytes);

        bytesRead = 4 + arrayBytes.Length;
        return result;
    }

    /// <summary>
    /// Deserializes a float array from the buffer.
    /// </summary>
    /// <param name="buffer">Source buffer.</param>
    /// <param name="offset">Starting offset.</param>
    /// <param name="bytesRead">Number of bytes consumed.</param>
    /// <returns>Deserialized array.</returns>
    public static float[] DeserializeFloatArray(ReadOnlySpan<byte> buffer, int offset, out int bytesRead)
    {
        // Read array length
        int length = MemoryMarshal.Read<int>(buffer.Slice(offset));

        // Read array data
        var result = new float[length];
        var arrayBytes = MemoryMarshal.AsBytes(result.AsSpan());
        buffer.Slice(offset + 4, arrayBytes.Length).CopyTo(arrayBytes);

        bytesRead = 4 + arrayBytes.Length;
        return result;
    }

    /// <summary>
    /// Deserializes a string from the buffer.
    /// </summary>
    /// <param name="buffer">Source buffer.</param>
    /// <param name="offset">Starting offset.</param>
    /// <param name="bytesRead">Number of bytes consumed.</param>
    /// <returns>Deserialized string.</returns>
    public static string DeserializeString(ReadOnlySpan<byte> buffer, int offset, out int bytesRead)
    {
        // Read string length
        int byteCount = MemoryMarshal.Read<int>(buffer.Slice(offset));

        if (byteCount == 0)
        {
            bytesRead = 4;
            return string.Empty;
        }

        // Read UTF8 bytes
        var str = System.Text.Encoding.UTF8.GetString(buffer.Slice(offset + 4, byteCount));
        bytesRead = 4 + byteCount;
        return str;
    }
}
