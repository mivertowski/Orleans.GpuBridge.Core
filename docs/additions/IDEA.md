Love it. Turning **grains into native GPU actors** (each grain = a persistent kernel) is absolutely doable—and spicy. Here’s how I’d shape it.

# Concept: K-Grains (GPU-resident grains)

* **What:** A K-Grain is a **persistent kernel**+state that runs on the GPU and processes **messages** from a device-side mailbox.
* **Why:** Ultra-low latency, zero launch overhead, device-resident state, and GPU↔GPU dataflow without bouncing through CPU.

## Core pieces

1. **K-Runtime (on GPU):**

   * One resident kernel grid per device (or per partition) that multiplexes **thousands of K-Grains**.
   * Each K-Grain gets:

     * a **mailbox**: lock-free ring buffer in global memory,
     * a **state slot** in device memory,
     * optional **timer wheel** for periodic work.
   * Warps act as **mailbox pollers**; work is executed warp-cooperatively to keep occupancy high.

2. **Host Bridge (Orleans side):**

   * **KDirectory**: maps `KGrainId → {device, mailbox ptr, state ptr}`.
   * **KRouter**: routes messages:

     * intra-device: direct enqueue to device mailbox,
     * inter-device (same node): P2P copy or host-mediated enqueue,
     * inter-silo (network): Orleans stream → target silo → device enqueue.
   * **Activation** of a K-Grain = allocate device state + register mailbox + (optionally) schedule a block/warp.

3. **Messaging**

   * **Message envelope (fixed 128B)**: `{dst, src, type, seq, payload_len, payload_ptr/inline[...]}`
   * **Intra-device:** atomics on head/tail; memory fences; no CPU copies.
   * **Inter-device same node:** host‐mediated enqueue, with peer access if available.
   * **Cross-silo:** serialize to pinned host memory → network → target → device enqueue.

4. **Reliability modes**

   * **At-most-once** (default): simplest, fastest.
   * **At-least-once:** ack messages + resend on timeout (host router).
   * **Exactly-once:** add `seq` + device-side dedupe window (LRU/bitmap) + **epoch** to bound state.

5. **State, snapshots, audit**

   * Device-resident **event-sourced** state deltas buffered in mapped memory.
   * Async snapshot to host per **epoch** or timer; replay is always host-driven → deterministic.

6. **Security/multi-tenant**

   * Partition GPU memory into **tenancy segments**; never share mailboxes across tenants.
   * On A/H-class hardware, use **MIG** (if available) to hard-partition compute.

---

## Minimal APIs (host side)

```csharp
// Orleans-facing K-Grain
public interface IKGrain<TMsg> : IGrainWithStringKey
{
    Task SendAsync(TMsg msg, KSendOptions? opts = null);
    Task<KGrainInfo> GetInfoAsync();
}

public record KSendOptions(string? PreferredDevice = null, bool HighPriority = false);

// System grain for routing
public interface IKRouter : IGrainWithIntegerKey
{
    Task RouteAsync(KEnvelope env);              // cross-device/silo
    Task<KGrainInfo> RegisterAsync(KGrainInfo i);
}
```

### Dev ergonomics (what you implement)

```csharp
[KGrain("kernels/P2P_Motif")]
public sealed class P2PMotifKGrain : KGrainBase<MotifMsg, MotifState>
{
    // Runs on GPU (compiled via DotCompute): OnMessage() gets inlined into the resident kernel loop
    [GpuEntry] 
    public static void OnMessage(ref MotifState state, in MotifMsg msg, KContext ctx)
    {
        // update device-resident state, push to neighbors, emit metrics, etc.
    }
}
```

---

## Device memory layout (intra-device)

* **Mailboxes:** `N` ring buffers (power-of-2 sized), each with `{head, tail, slots[]}`.
* **State area:** SoA/structure-of-arrays per type for coalesced access.
* **Completion/metrics:** mapped buffers polled by host for observability.

```
| Mailboxes (N) | States (N * sizeof(T)) | Timer Wheel | Metrics |
```

---

## Kernel loop (sketch, CUDA-ish pseudocode)

```cpp
__global__ void ResidentKernel(DeviceCtx* ctx) {
  const int kgrain = blockIdx.x;         // one block per K-Grain (or per K-Grain group)
  auto& mb = ctx->mailboxes[kgrain];
  auto& st = ctx->states[kgrain];

  while (__ldg(&ctx->running)) {
    int has = mb_try_pop(mb, &msg);      // lock-free pop
    if (has) {
      OnMessage(st, msg, ctx);           // generated from user code
    } else {
      // timers/heartbeats, then polite backoff
      __nanosleep(ctx->backoff_ns);
    }
  }
}
```

> Scale trick: instead of 1 block per K-Grain (expensive), run **a fixed grid** and schedule many K-Grains through a **device work queue**. Each work item points to a mailbox needing service. That way thousands of K-Grains share a smaller number of blocks.

---

## Inter-K-Grain messaging

* **Same device:** `enqueue(dst_mailbox, msg)` (device function).
* **Different device (same node):** host **fast-path**: copy envelope to target device’s mailbox region with `cudaMemcpyPeerAsync` (or equivalent).
* **Different silo:** Orleans stream to `IKRouter` on target → enqueue.
* Optional **NVSHMEM/NCCL** later for GPU-initiated inter-device sends if you want to bypass the host entirely (advanced).

---

## Backpressure & scheduling

* **Mailbox watermarks** (hi/lo): backpressure the sender via host router; drop or route to overflow buffer when over hi-watermark (configurable).
* **Priority lanes:** high-priority mailboxes are polled more frequently; host sets **fair-share** across tenants.
* **Auto-tuning (Pro):** adjust **micro-batch**/work-stealing and **backoff** dynamically to hit p95 targets.

---

## Orleans integration

* **Placement:** new `KPlacementStrategy` that prefers GPU-capable silos with free slots for the K-Grain’s device partition.
* **Activation:** K-Grain activation allocates mailbox + state on device and registers with **KDirectory**; deactivation flushes snapshot.
* **Interop:** regular CPU grains can `SendAsync` to K-Grains; K-Grains can **emit Orleans events** via mapped buffer → host reader → Orleans stream.

---

## MVP plan (2–3 sprints)

**Sprint 1**

* Intra-device only: device mailboxes, resident kernel loop, host enqueue/dequeue; Orleans `IKGrain<T>` façade + `KDirectory`.
* Exactly-once not required—**at-most-once** with counters and snapshots.

**Sprint 2**

* Inter-device (same silo): host-mediated peer copies; backpressure, priorities; timer wheel; perf counters.
* Determinism via **epoch** + state snapshots.

**Sprint 3**

* Cross-silo routing; **at-least-once** (acks/retries); basic **auto-tuner** for poll/backoff; metrics exporter.

---

## Where to put it in your repos

* **Core (OSS):** `Orleans.KGpu` with K-Runtime stubs, KDirectory/KRouter, intra-device messaging, Orleans façade, CPU fallback.
* **Pro:** inter-device fast-path, cross-silo optimizations, **auto-tuning**, advanced kernels (motif/JE/OCEL), dashboards.

---

## Gotchas (so we don’t get bitten)

* **SM starvation:** Persistent kernels hog the device—keep **grid size bounded** and multiplex K-Grains through work queues.
* **Memory pressure:** per-grain mailboxes add up—pool buffers, allow **shared queues** with routing headers.
* **Determinism:** floating-point order can vary; log control messages, not raw FP results, or use **deterministic reductions** where needed.
* **Debugging:** provide a host **single-step mode** (CPU backend) and a **device trace buffer**.

---

If you want, I can:

* generate a **repo skeleton** `Orleans.KGpu` with the **resident kernel loop**, device mailbox structs, host router, Orleans façades, and a tiny **“source → add → sink”** K-Grain pipeline sample;
* include **bench scaffolding** (msg/sec, p95/99, occupancy) so you can tune quickly.

Say the word and I’ll package it as a downloadable starter.
