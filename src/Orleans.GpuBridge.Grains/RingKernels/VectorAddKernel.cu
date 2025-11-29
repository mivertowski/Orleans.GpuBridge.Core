// Copyright (c) 2025 Michael Ivertowski
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// GPU Ring Kernel for VectorAddActor
// This kernel runs persistently on GPU, processing messages from Orleans actors

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/atomic>

namespace cg = cooperative_groups;

// Message queue structure (lock-free ring buffer)
template<typename T>
struct MessageQueue {
    T* buffer;                      // Message buffer in GPU memory
    int capacity;                   // Queue capacity (power of 2)
    cuda::atomic<int>* head;        // Dequeue position (atomic)
    cuda::atomic<int>* tail;        // Enqueue position (atomic)

    __device__ bool try_dequeue(T& message) {
        int current_head = head->load(cuda::memory_order_relaxed);
        int current_tail = tail->load(cuda::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Queue empty
        }

        int next_head = (current_head + 1) & (capacity - 1);

        // Try to claim slot with CAS
        if (head->compare_exchange_strong(current_head, next_head,
                                           cuda::memory_order_release,
                                           cuda::memory_order_relaxed)) {
            message = buffer[current_head];
            return true;
        }
        return false; // Lost race with another thread
    }

    __device__ bool try_enqueue(const T& message) {
        int current_tail = tail->load(cuda::memory_order_relaxed);
        int next_tail = (current_tail + 1) & (capacity - 1);
        int current_head = head->load(cuda::memory_order_acquire);

        if (next_tail == current_head) {
            return false; // Queue full
        }

        // Try to claim slot with CAS
        if (tail->compare_exchange_strong(current_tail, next_tail,
                                           cuda::memory_order_release,
                                           cuda::memory_order_relaxed)) {
            buffer[current_tail] = message;
            return true;
        }
        return false; // Lost race
    }
};

// Kernel control structure (64 bytes, cache-aligned)
struct KernelControl {
    cuda::atomic<int> active;          // 1 = active, 0 = paused
    cuda::atomic<int> terminate;       // 1 = terminate requested
    cuda::atomic<int> terminated;      // 1 = kernel has exited
    cuda::atomic<int> errors;          // Error counter
    cuda::atomic<long> msg_count;      // Messages processed
    cuda::atomic<long> last_activity;  // Timestamp of last activity
    long input_queue_head;             // Input queue head pointer
    long input_queue_tail;             // Input queue tail pointer
    long output_queue_head;            // Output queue head pointer
    long output_queue_tail;            // Output queue tail pointer
};

// OrleansGpuMessage (256 bytes) - must match C# struct
struct OrleansGpuMessage {
    int method_id;                     // Method hash
    long timestamp_ticks;              // UTC ticks
    long correlation_id;               // Request/response matching
    int type;                          // MessageType enum
    int sender_id;                     // Sender actor ID
    int target_id;                     // Target actor ID
    int reserved1;
    int reserved2;
    char payload[228];                 // Inline payload
};

// VectorAddRequest (must match C# struct)
struct VectorAddRequest {
    int vector_a_length;
    int vector_b_length;
    int operation;                     // VectorOperation enum
    int reserved;
    float inline_data_a[25];           // First 25 elements of vector A
    float inline_data_b[25];           // First 25 elements of vector B
};

// VectorAddResponse (must match C# struct)
struct VectorAddResponse {
    int result_length;
    float scalar_result;
    long reserved;
    float inline_result[50];
};

// Persistent ring kernel for VectorAddActor
extern "C" __global__ void __launch_bounds__(256, 2) VectorAddActor_kernel(
    MessageQueue<OrleansGpuMessage>* input_queue,
    MessageQueue<OrleansGpuMessage>* output_queue,
    KernelControl* control,
    float* workspace,                  // Shared workspace for vector operations
    int workspace_size)
{
    // Get thread and block IDs
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Cooperative groups for synchronization
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Thread 0 handles message I/O, others do computation
    const bool is_io_thread = (tid == 0 && bid == 0);

    // Persistent kernel loop (runs forever until terminated)
    while (true) {
        // Check for termination request
        if (control->terminate.load(cuda::memory_order_acquire) == 1) {
            if (is_io_thread) {
                control->terminated.store(1, cuda::memory_order_release);
            }
            break;
        }

        // Wait for activation
        while (control->active.load(cuda::memory_order_acquire) == 0) {
            if (control->terminate.load(cuda::memory_order_acquire) == 1) {
                if (is_io_thread) {
                    control->terminated.store(1, cuda::memory_order_release);
                }
                return;
            }
            __nanosleep(1000); // Sleep 1 microsecond
        }

        // Process messages (I/O thread dequeues, all threads compute)
        OrleansGpuMessage msg;
        bool has_message = false;

        if (is_io_thread) {
            has_message = input_queue->try_dequeue(msg);
        }

        // Broadcast has_message to all threads in block
        has_message = __syncthreads_or(has_message);

        if (has_message) {
            // Deserialize request from payload
            VectorAddRequest* req = (VectorAddRequest*)msg.payload;

            // Determine operation
            int operation = req->operation;
            int length = req->vector_a_length;

            // Allocate shared memory for result (if small enough)
            __shared__ float shared_result[256];

            // Perform vector addition (parallel across threads)
            if (tid < length && length <= 25) {
                // Small vectors - use inline data
                float a = req->inline_data_a[tid];
                float b = req->inline_data_b[tid];
                shared_result[tid] = a + b;
            }

            // Synchronize threads
            __syncthreads();

            // Create response message
            if (is_io_thread) {
                OrleansGpuMessage response;
                response.method_id = msg.method_id;
                response.timestamp_ticks = msg.timestamp_ticks;
                response.correlation_id = msg.correlation_id;
                response.type = 1; // MessageType.Response
                response.sender_id = msg.target_id; // Swap sender/target
                response.target_id = msg.sender_id;

                VectorAddResponse* resp = (VectorAddResponse*)response.payload;
                resp->result_length = length;

                if (operation == 0) {
                    // VectorOperation.Add - return full vector
                    for (int i = 0; i < length && i < 50; i++) {
                        resp->inline_result[i] = shared_result[i];
                    }
                    resp->scalar_result = 0.0f;
                } else if (operation == 1) {
                    // VectorOperation.AddScalar - return scalar sum
                    float sum = 0.0f;
                    for (int i = 0; i < length && i < 25; i++) {
                        sum += shared_result[i];
                    }
                    resp->scalar_result = sum;
                    resp->result_length = 1;
                }

                // Enqueue response (with retry logic)
                bool enqueued = false;
                for (int retry = 0; retry < 100 && !enqueued; retry++) {
                    enqueued = output_queue->try_enqueue(response);
                    if (!enqueued) {
                        __nanosleep(100); // Wait 100ns before retry
                    }
                }

                if (!enqueued) {
                    // Failed to enqueue response - increment error counter
                    control->errors.fetch_add(1, cuda::memory_order_relaxed);
                } else {
                    // Success - update message counter and activity timestamp
                    control->msg_count.fetch_add(1, cuda::memory_order_relaxed);
                    control->last_activity.store(
                        clock64(),
                        cuda::memory_order_relaxed);
                }
            }
        } else {
            // No message available - yield to other warps
            __nanosleep(100); // Sleep 100ns before checking again
        }

        // Grid synchronization (ensures all blocks are in sync)
        // Only needed for multi-block kernels
        if (gridDim.x > 1) {
            grid.sync();
        }
    }
}

// Host-side kernel launcher (called from C#)
extern "C"
cudaError_t launch_vector_add_actor(
    MessageQueue<OrleansGpuMessage>* input_queue,
    MessageQueue<OrleansGpuMessage>* output_queue,
    KernelControl* control,
    float* workspace,
    int workspace_size,
    int grid_size,
    int block_size,
    cudaStream_t stream)
{
    // Launch persistent kernel with cooperative groups
    void* args[] = {
        &input_queue,
        &output_queue,
        &control,
        &workspace,
        &workspace_size
    };

    return cudaLaunchCooperativeKernel(
        (void*)VectorAddActor_kernel,
        dim3(grid_size),
        dim3(block_size),
        args,
        0,  // Shared memory size (using __shared__ instead)
        stream);
}
