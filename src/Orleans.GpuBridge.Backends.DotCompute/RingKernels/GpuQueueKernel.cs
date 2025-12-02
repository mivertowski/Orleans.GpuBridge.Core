// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Backends.DotCompute.Attributes;

namespace Orleans.GpuBridge.Backends.DotCompute.RingKernels;

/// <summary>
/// GPU kernel for atomic queue operations enabling lock-free K2K messaging.
/// </summary>
/// <remarks>
/// <para>
/// This kernel provides GPU-native atomic operations for queue management:
/// <list type="bullet">
/// <item><description>atomicAdd for tail increment (claiming write slots)</description></item>
/// <item><description>atomicCAS for completion flag updates</description></item>
/// <item><description>Memory fences for visibility between writers/readers</description></item>
/// </list>
/// </para>
/// <para>
/// Queue memory layout (GPU unified memory):
/// <code>
/// [WriteIndex:4][ReadIndex:4][WriteCommitted:4][Capacity:4][MessageSize:4][Reserved:12][Data...]
/// </code>
/// </para>
/// <para>
/// For GPUs without hostNativeAtomicSupported (like RTX series), this provides
/// a batched EventDriven mode with GPU-side atomics for queue coherence.
/// </para>
/// </remarks>
public sealed class GpuQueueKernel : IDisposable
{
    private readonly ILogger<GpuQueueKernel> _logger;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Queue header offset constants (matches RingKernelQueueHeader layout).
    /// </summary>
    private static class QueueOffsets
    {
        public const int WriteIndex = 0;
        public const int ReadIndex = 4;
        public const int WriteCommitted = 8;
        public const int Capacity = 12;
        public const int MessageSize = 16;
        public const int HeaderSize = 32;
    }

    /// <summary>
    /// GPU kernel source for atomic queue operations (CUDA).
    /// </summary>
    /// <remarks>
    /// Uses CUDA atomicAdd for lock-free tail increment and
    /// __threadfence() for memory visibility across GPU threads.
    /// </remarks>
    private const string CudaQueueKernelSource = @"
extern ""C"" {

// Queue header structure (must match RingKernelQueueHeader)
struct QueueHeader {
    int writeIndex;
    int readIndex;
    int writeCommitted;
    int capacity;
    int messageSize;
    int reserved[3];
};

// Atomically enqueue a message to the GPU queue
// Returns the slot index where the message should be written, or -1 if queue is full
__device__ int atomicEnqueue(volatile QueueHeader* header) {
    // Atomically increment write index to claim a slot
    int claimed = atomicAdd((int*)&header->writeIndex, 1);

    // Check if queue is full (claimed slot would wrap past read index)
    int readIdx = header->readIndex;
    if (claimed - readIdx >= header->capacity) {
        // Queue full - decrement write index to release claimed slot
        atomicSub((int*)&header->writeIndex, 1);
        return -1;
    }

    // Return slot index (using modulo for ring buffer wrap)
    return claimed & (header->capacity - 1);
}

// Signal that a message write is complete
__device__ void atomicCommitWrite(volatile QueueHeader* header) {
    // Memory fence to ensure message data is visible before commit
    __threadfence();

    // Atomically increment writeCommitted
    atomicAdd((int*)&header->writeCommitted, 1);

    // Another fence to ensure commit is visible
    __threadfence();
}

// Atomically dequeue a message from the GPU queue
// Returns the slot index to read from, or -1 if queue is empty
__device__ int atomicDequeue(volatile QueueHeader* header) {
    // Check if messages are available
    if (header->writeCommitted <= header->readIndex) {
        return -1; // Queue empty
    }

    // Atomically increment read index to claim a message
    int claimed = atomicAdd((int*)&header->readIndex, 1);

    // Verify the claimed slot was valid (could race with another reader)
    if (claimed >= header->writeCommitted) {
        // No message available at this slot - release
        atomicSub((int*)&header->readIndex, 1);
        return -1;
    }

    // Return slot index
    return claimed & (header->capacity - 1);
}

// Kernel: Enqueue a single message (CPU-launched for partial coherence GPUs)
__global__ void enqueue_message(
    volatile QueueHeader* header,
    unsigned char* messageData,
    int messageSizeBytes,
    int* resultSlot)
{
    int slot = atomicEnqueue(header);

    if (slot >= 0) {
        // Calculate write position
        unsigned char* dataStart = (unsigned char*)(header + 1);
        unsigned char* writePtr = dataStart + (slot * header->messageSize);

        // Copy message data (single thread for simplicity)
        for (int i = 0; i < messageSizeBytes && i < header->messageSize; i++) {
            writePtr[i] = messageData[i];
        }

        // Commit the write
        atomicCommitWrite(header);
    }

    *resultSlot = slot;
}

// Kernel: Dequeue a single message (CPU-launched for partial coherence GPUs)
__global__ void dequeue_message(
    volatile QueueHeader* header,
    unsigned char* messageBuffer,
    int bufferSizeBytes,
    int* resultSlot)
{
    int slot = atomicDequeue(header);

    if (slot >= 0) {
        // Memory fence to ensure we see the latest message data
        __threadfence();

        // Calculate read position
        unsigned char* dataStart = (unsigned char*)(header + 1);
        unsigned char* readPtr = dataStart + (slot * header->messageSize);

        // Copy message data
        int copySize = min(bufferSizeBytes, header->messageSize);
        for (int i = 0; i < copySize; i++) {
            messageBuffer[i] = readPtr[i];
        }
    }

    *resultSlot = slot;
}

// Kernel: Write response to K2K response slot
__global__ void write_k2k_response(
    long responsePtr,
    unsigned char* responseData,
    int responseSizeBytes)
{
    // Response slot layout: [CompletionFlag:4 bytes][Response data...]
    volatile int* completionFlag = (volatile int*)responsePtr;
    unsigned char* dataPtr = (unsigned char*)responsePtr + 4;

    // Write response data first
    for (int i = 0; i < responseSizeBytes; i++) {
        dataPtr[i] = responseData[i];
    }

    // Memory barrier to ensure data is visible
    __threadfence_system();

    // Set completion flag to signal response is ready (1 = ready)
    atomicExch((int*)completionFlag, 1);
}

// Kernel: Batch enqueue multiple messages (more efficient for high throughput)
__global__ void batch_enqueue(
    volatile QueueHeader* header,
    unsigned char* messages,
    int messageCount,
    int messageSizeBytes,
    int* successCount)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < messageCount) {
        int slot = atomicEnqueue(header);

        if (slot >= 0) {
            // Calculate positions
            unsigned char* srcPtr = messages + (tid * messageSizeBytes);
            unsigned char* dataStart = (unsigned char*)(header + 1);
            unsigned char* dstPtr = dataStart + (slot * header->messageSize);

            // Copy message
            for (int i = 0; i < messageSizeBytes && i < header->messageSize; i++) {
                dstPtr[i] = srcPtr[i];
            }

            // Commit
            atomicCommitWrite(header);

            // Increment success count
            atomicAdd(successCount, 1);
        }
    }
}

// Kernel: Get queue status (non-atomic read for monitoring)
__global__ void get_queue_status(
    volatile QueueHeader* header,
    int* writeIndex,
    int* readIndex,
    int* writeCommitted,
    int* available,
    int* freeSlots)
{
    *writeIndex = header->writeIndex;
    *readIndex = header->readIndex;
    *writeCommitted = header->writeCommitted;
    *available = header->writeCommitted - header->readIndex;
    *freeSlots = header->capacity - (header->writeIndex - header->readIndex);
}

} // extern ""C""
";

    /// <summary>
    /// GPU kernel source for atomic queue operations (OpenCL).
    /// </summary>
    /// <remarks>
    /// Uses OpenCL atomic_add for lock-free operations and
    /// mem_fence for memory visibility across work items.
    /// </remarks>
    private const string OpenClQueueKernelSource = @"
// Queue header structure (must match RingKernelQueueHeader)
typedef struct {
    int writeIndex;
    int readIndex;
    int writeCommitted;
    int capacity;
    int messageSize;
    int reserved[3];
} QueueHeader;

// Atomically enqueue a message to the GPU queue
int atomicEnqueue(__global volatile QueueHeader* header) {
    // Atomically increment write index to claim a slot
    int claimed = atomic_add((__global volatile int*)&header->writeIndex, 1);

    // Check if queue is full
    int readIdx = header->readIndex;
    if (claimed - readIdx >= header->capacity) {
        atomic_sub((__global volatile int*)&header->writeIndex, 1);
        return -1;
    }

    return claimed & (header->capacity - 1);
}

// Signal that a message write is complete
void atomicCommitWrite(__global volatile QueueHeader* header) {
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    atomic_add((__global volatile int*)&header->writeCommitted, 1);
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

// Kernel: Enqueue a single message
__kernel void enqueue_message(
    __global volatile QueueHeader* header,
    __global uchar* messageData,
    int messageSizeBytes,
    __global int* resultSlot)
{
    int slot = atomicEnqueue(header);

    if (slot >= 0) {
        __global uchar* dataStart = (__global uchar*)(header + 1);
        __global uchar* writePtr = dataStart + (slot * header->messageSize);

        for (int i = 0; i < messageSizeBytes && i < header->messageSize; i++) {
            writePtr[i] = messageData[i];
        }

        atomicCommitWrite(header);
    }

    *resultSlot = slot;
}

// Kernel: Dequeue a single message
__kernel void dequeue_message(
    __global volatile QueueHeader* header,
    __global uchar* messageBuffer,
    int bufferSizeBytes,
    __global int* resultSlot)
{
    if (header->writeCommitted <= header->readIndex) {
        *resultSlot = -1;
        return;
    }

    int claimed = atomic_add((__global volatile int*)&header->readIndex, 1);

    if (claimed >= header->writeCommitted) {
        atomic_sub((__global volatile int*)&header->readIndex, 1);
        *resultSlot = -1;
        return;
    }

    int slot = claimed & (header->capacity - 1);

    mem_fence(CLK_GLOBAL_MEM_FENCE);

    __global uchar* dataStart = (__global uchar*)(header + 1);
    __global uchar* readPtr = dataStart + (slot * header->messageSize);

    int copySize = min(bufferSizeBytes, header->messageSize);
    for (int i = 0; i < copySize; i++) {
        messageBuffer[i] = readPtr[i];
    }

    *resultSlot = slot;
}

// Kernel: Write response to K2K response slot
__kernel void write_k2k_response(
    long responsePtr,
    __global uchar* responseData,
    int responseSizeBytes)
{
    __global volatile int* completionFlag = (__global volatile int*)responsePtr;
    __global uchar* dataPtr = (__global uchar*)responsePtr + 4;

    for (int i = 0; i < responseSizeBytes; i++) {
        dataPtr[i] = responseData[i];
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    atomic_xchg(completionFlag, 1);
}

// Kernel: Get queue status
__kernel void get_queue_status(
    __global volatile QueueHeader* header,
    __global int* writeIndex,
    __global int* readIndex,
    __global int* writeCommitted,
    __global int* available,
    __global int* freeSlots)
{
    *writeIndex = header->writeIndex;
    *readIndex = header->readIndex;
    *writeCommitted = header->writeCommitted;
    *available = header->writeCommitted - header->readIndex;
    *freeSlots = header->capacity - (header->writeIndex - header->readIndex);
}
";

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuQueueKernel"/> class.
    /// </summary>
    /// <param name="logger">Logger for kernel operations.</param>
    public GpuQueueKernel(ILogger<GpuQueueKernel> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Gets the CUDA kernel source code.
    /// </summary>
    public static string CudaSource => CudaQueueKernelSource;

    /// <summary>
    /// Gets the OpenCL kernel source code.
    /// </summary>
    public static string OpenClSource => OpenClQueueKernelSource;

    /// <summary>
    /// Enqueues a message to the GPU queue using CPU-side atomic operations.
    /// </summary>
    /// <remarks>
    /// This is the CPU fallback for GPUs without full coherence support.
    /// Uses Interlocked operations that are compatible with GPU unified memory.
    /// </remarks>
    /// <typeparam name="TMessage">The message type (must be unmanaged).</typeparam>
    /// <param name="queuePtr">Pointer to the queue memory.</param>
    /// <param name="message">The message to enqueue.</param>
    /// <returns>The slot index where the message was written, or -1 if queue is full.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe int EnqueueWithAtomics<TMessage>(IntPtr queuePtr, ref TMessage message)
        where TMessage : unmanaged
    {
        if (queuePtr == IntPtr.Zero)
        {
            return -1;
        }

        var header = (int*)queuePtr;
        var capacity = header[QueueOffsets.Capacity / 4];
        var messageSize = header[QueueOffsets.MessageSize / 4];

        if (capacity <= 0 || messageSize <= 0)
        {
            _logger.LogWarning("Invalid queue header: capacity={Capacity}, messageSize={MessageSize}",
                capacity, messageSize);
            return -1;
        }

        // Atomically claim a write slot
        var claimed = Interlocked.Increment(ref header[QueueOffsets.WriteIndex / 4]) - 1;
        var readIndex = Volatile.Read(ref header[QueueOffsets.ReadIndex / 4]);

        // Check if queue is full
        if (claimed - readIndex >= capacity)
        {
            // Release the claimed slot
            Interlocked.Decrement(ref header[QueueOffsets.WriteIndex / 4]);
            _logger.LogTrace("Queue full: claimed={Claimed}, readIndex={ReadIndex}, capacity={Capacity}",
                claimed, readIndex, capacity);
            return -1;
        }

        // Calculate write position (power-of-2 capacity allows bitwise AND)
        var slotIndex = claimed & (capacity - 1);
        var dataStart = (byte*)queuePtr + QueueOffsets.HeaderSize;
        var writePtr = dataStart + (slotIndex * messageSize);

        // Copy message data
        var msgSize = sizeof(TMessage);
        var copySize = Math.Min(msgSize, messageSize);
        Unsafe.CopyBlockUnaligned(writePtr, Unsafe.AsPointer(ref message), (uint)copySize);

        // Memory barrier to ensure message is visible before commit
        Thread.MemoryBarrier();

        // Commit the write
        Interlocked.Increment(ref header[QueueOffsets.WriteCommitted / 4]);

        _logger.LogTrace("Enqueued message to slot {SlotIndex} (size={Size} bytes)",
            slotIndex, copySize);

        return slotIndex;
    }

    /// <summary>
    /// Dequeues a message from the GPU queue using CPU-side atomic operations.
    /// </summary>
    /// <typeparam name="TMessage">The message type (must be unmanaged).</typeparam>
    /// <param name="queuePtr">Pointer to the queue memory.</param>
    /// <param name="message">The dequeued message (if successful).</param>
    /// <returns>True if a message was dequeued, false if queue is empty.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe bool DequeueWithAtomics<TMessage>(IntPtr queuePtr, out TMessage message)
        where TMessage : unmanaged
    {
        message = default;

        if (queuePtr == IntPtr.Zero)
        {
            return false;
        }

        var header = (int*)queuePtr;
        var capacity = header[QueueOffsets.Capacity / 4];
        var messageSize = header[QueueOffsets.MessageSize / 4];

        // Check if messages are available
        var writeCommitted = Volatile.Read(ref header[QueueOffsets.WriteCommitted / 4]);
        var readIndex = Volatile.Read(ref header[QueueOffsets.ReadIndex / 4]);

        if (writeCommitted <= readIndex)
        {
            return false; // Queue empty
        }

        // Atomically claim a message
        var claimed = Interlocked.Increment(ref header[QueueOffsets.ReadIndex / 4]) - 1;

        // Verify we got a valid message
        writeCommitted = Volatile.Read(ref header[QueueOffsets.WriteCommitted / 4]);
        if (claimed >= writeCommitted)
        {
            // No message at this slot - release
            Interlocked.Decrement(ref header[QueueOffsets.ReadIndex / 4]);
            return false;
        }

        // Memory barrier to ensure we see latest message data
        Thread.MemoryBarrier();

        // Calculate read position
        var slotIndex = claimed & (capacity - 1);
        var dataStart = (byte*)queuePtr + QueueOffsets.HeaderSize;
        var readPtr = dataStart + (slotIndex * messageSize);

        // Copy message data
        var msgSize = sizeof(TMessage);
        var copySize = Math.Min(msgSize, messageSize);
        message = Unsafe.ReadUnaligned<TMessage>(readPtr);

        _logger.LogTrace("Dequeued message from slot {SlotIndex}", slotIndex);

        return true;
    }

    /// <summary>
    /// Gets the current queue status.
    /// </summary>
    /// <param name="queuePtr">Pointer to the queue memory.</param>
    /// <returns>Queue status information.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe QueueStatus GetQueueStatus(IntPtr queuePtr)
    {
        if (queuePtr == IntPtr.Zero)
        {
            return new QueueStatus { IsValid = false };
        }

        var header = (int*)queuePtr;

        var writeIndex = Volatile.Read(ref header[QueueOffsets.WriteIndex / 4]);
        var readIndex = Volatile.Read(ref header[QueueOffsets.ReadIndex / 4]);
        var writeCommitted = Volatile.Read(ref header[QueueOffsets.WriteCommitted / 4]);
        var capacity = header[QueueOffsets.Capacity / 4];
        var messageSize = header[QueueOffsets.MessageSize / 4];

        return new QueueStatus
        {
            IsValid = true,
            WriteIndex = writeIndex,
            ReadIndex = readIndex,
            WriteCommitted = writeCommitted,
            Capacity = capacity,
            MessageSize = messageSize,
            AvailableMessages = writeCommitted - readIndex,
            FreeSlots = capacity - (writeIndex - readIndex)
        };
    }

    /// <summary>
    /// Initializes a queue header at the specified memory location.
    /// </summary>
    /// <param name="queuePtr">Pointer to the queue memory.</param>
    /// <param name="capacity">Queue capacity (must be power of 2).</param>
    /// <param name="messageSize">Size of each message slot in bytes.</param>
    public unsafe void InitializeQueue(IntPtr queuePtr, int capacity, int messageSize)
    {
        if (queuePtr == IntPtr.Zero)
        {
            throw new ArgumentException("Queue pointer cannot be zero", nameof(queuePtr));
        }

        if (capacity <= 0 || (capacity & (capacity - 1)) != 0)
        {
            throw new ArgumentException("Capacity must be a positive power of 2", nameof(capacity));
        }

        if (messageSize <= 0 || messageSize % 4 != 0)
        {
            throw new ArgumentException("Message size must be a positive multiple of 4", nameof(messageSize));
        }

        var header = (int*)queuePtr;

        // Initialize header atomically
        Volatile.Write(ref header[QueueOffsets.WriteIndex / 4], 0);
        Volatile.Write(ref header[QueueOffsets.ReadIndex / 4], 0);
        Volatile.Write(ref header[QueueOffsets.WriteCommitted / 4], 0);
        header[QueueOffsets.Capacity / 4] = capacity;
        header[QueueOffsets.MessageSize / 4] = messageSize;

        // Memory barrier to ensure initialization is visible
        Thread.MemoryBarrier();

        _logger.LogDebug("Initialized GPU queue: capacity={Capacity}, messageSize={MessageSize}",
            capacity, messageSize);
    }

    /// <summary>
    /// Writes a response to a K2K response slot using atomic operations.
    /// </summary>
    /// <typeparam name="TResponse">The response type (must be unmanaged).</typeparam>
    /// <param name="responsePtr">Pointer to the response slot.</param>
    /// <param name="response">The response data.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void WriteK2KResponse<TResponse>(IntPtr responsePtr, TResponse response)
        where TResponse : unmanaged
    {
        if (responsePtr == IntPtr.Zero)
        {
            return;
        }

        // Response slot layout: [CompletionFlag:4 bytes][Response data...]
        var completionFlag = (int*)responsePtr;
        var dataPtr = (byte*)responsePtr + 4;

        // Write response data first
        Unsafe.WriteUnaligned(dataPtr, response);

        // Memory barrier to ensure response data is visible before completion flag
        Thread.MemoryBarrier();

        // Set completion flag atomically (1 = ready)
        Volatile.Write(ref *completionFlag, 1);
    }

    /// <summary>
    /// Checks if a K2K response is ready.
    /// </summary>
    /// <param name="responsePtr">Pointer to the response slot.</param>
    /// <returns>True if response is ready, false otherwise.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe bool IsK2KResponseReady(IntPtr responsePtr)
    {
        if (responsePtr == IntPtr.Zero)
        {
            return false;
        }

        return Volatile.Read(ref *(int*)responsePtr) == 1;
    }

    /// <summary>
    /// Reads a K2K response from a response slot.
    /// </summary>
    /// <typeparam name="TResponse">The response type (must be unmanaged).</typeparam>
    /// <param name="responsePtr">Pointer to the response slot.</param>
    /// <returns>The response data.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe TResponse ReadK2KResponse<TResponse>(IntPtr responsePtr)
        where TResponse : unmanaged
    {
        // Response slot layout: [CompletionFlag:4 bytes][Response data...]
        var dataPtr = (byte*)responsePtr + 4;
        return Unsafe.ReadUnaligned<TResponse>(dataPtr);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _logger.LogDebug("GPU queue kernel disposed");
    }
}

/// <summary>
/// Status information for a GPU queue.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct QueueStatus
{
    /// <summary>
    /// Whether the queue pointer is valid.
    /// </summary>
    public bool IsValid;

    /// <summary>
    /// Current write index (next position to write to).
    /// </summary>
    public int WriteIndex;

    /// <summary>
    /// Current read index (next position to read from).
    /// </summary>
    public int ReadIndex;

    /// <summary>
    /// Number of messages fully committed and ready to read.
    /// </summary>
    public int WriteCommitted;

    /// <summary>
    /// Maximum queue capacity.
    /// </summary>
    public int Capacity;

    /// <summary>
    /// Size of each message slot in bytes.
    /// </summary>
    public int MessageSize;

    /// <summary>
    /// Number of messages available for reading.
    /// </summary>
    public int AvailableMessages;

    /// <summary>
    /// Number of free slots available for writing.
    /// </summary>
    public int FreeSlots;
}
