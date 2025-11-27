// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Runtime.K2K;

/// <summary>
/// Header structure for ring kernel message queues enabling GPU-native K2K communication.
/// </summary>
/// <remarks>
/// <para>
/// This structure is laid out to match GPU memory alignment requirements and enables
/// lock-free atomic queue operations for sub-microsecond K2K message passing.
/// </para>
/// <para>
/// Memory layout:
/// <code>
/// [WriteIndex:4][ReadIndex:4][WriteCommitted:4][Capacity:4][MessageSize:4][Reserved:12][Data...]
/// </code>
/// Total header size: 32 bytes (aligned to GPU cache line for optimal performance)
/// </para>
/// <para>
/// Thread safety: All index fields are accessed via atomic operations.
/// Writers increment WriteIndex to claim a slot, write data, then update WriteCommitted.
/// Readers check WriteCommitted > ReadIndex before reading, then increment ReadIndex.
/// </para>
/// </remarks>
[StructLayout(LayoutKind.Explicit, Size = 32)]
public struct RingKernelQueueHeader
{
    /// <summary>
    /// Next write position (atomically incremented by writers to claim a slot).
    /// </summary>
    /// <remarks>
    /// Writers use Interlocked.Increment to atomically claim the next available slot.
    /// The actual slot index is computed as WriteIndex % Capacity.
    /// </remarks>
    [FieldOffset(0)]
    public int WriteIndex;

    /// <summary>
    /// Current read position (atomically incremented by readers after consuming a message).
    /// </summary>
    /// <remarks>
    /// Readers check (WriteCommitted > ReadIndex) to determine if messages are available,
    /// then use Interlocked.Increment to claim the next message to read.
    /// </remarks>
    [FieldOffset(4)]
    public int ReadIndex;

    /// <summary>
    /// Number of messages fully written and ready for consumption.
    /// </summary>
    /// <remarks>
    /// Writers update this after completing their write using a memory barrier.
    /// This provides the synchronization point between writers and readers.
    /// </remarks>
    [FieldOffset(8)]
    public int WriteCommitted;

    /// <summary>
    /// Maximum number of messages the queue can hold (must be power of 2).
    /// </summary>
    /// <remarks>
    /// Power of 2 allows using bitwise AND instead of modulo for index calculation.
    /// Queue is considered full when (WriteIndex - ReadIndex >= Capacity).
    /// </remarks>
    [FieldOffset(12)]
    public int Capacity;

    /// <summary>
    /// Size of each message slot in bytes (includes 4-byte message type prefix).
    /// </summary>
    /// <remarks>
    /// Each slot is: [MessageTypeId:4 bytes][Payload: MessageSize - 4 bytes]
    /// Must be a multiple of 4 for proper GPU memory alignment.
    /// </remarks>
    [FieldOffset(16)]
    public int MessageSize;

    /// <summary>
    /// Reserved field 1 for future use (padding to 32-byte alignment).
    /// </summary>
    [FieldOffset(20)]
    private int _reserved1;

    /// <summary>
    /// Reserved field 2 for future use (padding to 32-byte alignment).
    /// </summary>
    [FieldOffset(24)]
    private int _reserved2;

    /// <summary>
    /// Reserved field 3 for future use (padding to 32-byte alignment).
    /// </summary>
    [FieldOffset(28)]
    private int _reserved3;

    /// <summary>
    /// Creates a new queue header with the specified capacity and message size.
    /// </summary>
    /// <param name="capacity">Queue capacity (must be power of 2).</param>
    /// <param name="messageSize">Size of each message in bytes.</param>
    /// <returns>Initialized queue header.</returns>
    /// <exception cref="ArgumentException">If capacity is not a power of 2 or messageSize is invalid.</exception>
    public static RingKernelQueueHeader Create(int capacity, int messageSize)
    {
        if (capacity <= 0 || (capacity & (capacity - 1)) != 0)
        {
            throw new ArgumentException("Capacity must be a positive power of 2", nameof(capacity));
        }

        if (messageSize <= 0 || messageSize % 4 != 0)
        {
            throw new ArgumentException("Message size must be a positive multiple of 4", nameof(messageSize));
        }

        return new RingKernelQueueHeader
        {
            WriteIndex = 0,
            ReadIndex = 0,
            WriteCommitted = 0,
            Capacity = capacity,
            MessageSize = messageSize
        };
    }

    /// <summary>
    /// Gets the number of messages currently available for reading.
    /// </summary>
    /// <returns>Number of messages ready to be consumed.</returns>
    public readonly int GetAvailableCount()
    {
        return WriteCommitted - ReadIndex;
    }

    /// <summary>
    /// Gets the number of free slots available for writing.
    /// </summary>
    /// <returns>Number of slots available for new messages.</returns>
    public readonly int GetFreeCount()
    {
        return Capacity - (WriteIndex - ReadIndex);
    }

    /// <summary>
    /// Determines if the queue is full.
    /// </summary>
    /// <returns>True if no more messages can be enqueued.</returns>
    public readonly bool IsFull()
    {
        return WriteIndex - ReadIndex >= Capacity;
    }

    /// <summary>
    /// Determines if the queue is empty.
    /// </summary>
    /// <returns>True if no messages are available for reading.</returns>
    public readonly bool IsEmpty()
    {
        return WriteCommitted <= ReadIndex;
    }

    /// <summary>
    /// Calculates the total memory required for this queue (header + all message slots).
    /// </summary>
    /// <returns>Total bytes required for the queue allocation.</returns>
    public readonly int GetTotalAllocationSize()
    {
        return 32 + (Capacity * MessageSize);
    }

    /// <summary>
    /// Calculates the byte offset for a given slot index.
    /// </summary>
    /// <param name="slotIndex">The slot index (0 to Capacity-1).</param>
    /// <returns>Byte offset from the header start to the slot data.</returns>
    public readonly int GetSlotOffset(int slotIndex)
    {
        return 32 + (slotIndex * MessageSize);
    }

    /// <summary>
    /// Converts a write/read index to a slot index using modulo.
    /// </summary>
    /// <param name="index">The WriteIndex or ReadIndex value.</param>
    /// <returns>Slot index (0 to Capacity-1).</returns>
    public readonly int IndexToSlot(int index)
    {
        // Since Capacity is power of 2, use bitwise AND for faster modulo
        return index & (Capacity - 1);
    }
}
