# DotCompute Feature Request: MessageQueue Constructor Mismatch

**Date:** 2025-11-24
**Severity:** Bug / Breaking Issue
**Component:** `DotCompute.Backends.CUDA.RingKernels.CudaMessageQueueBridgeFactory`
**Affects:** Ring kernel message passing on CUDA backend

## Summary

The `CudaMessageQueueBridgeFactory.CreateNamedQueueAsync` method attempts to create a `MessageQueue<T>` instance with two parameters `(options, logger)`, but the `MessageQueue<T>` class only has a constructor that accepts one parameter `(MessageQueueOptions options)`.

## Error Observed

```
System.MissingMethodException: Constructor on type 'DotCompute.Core.Messaging.MessageQueue`1[[Orleans.GpuBridge.Backends.DotCompute.Temporal.VectorAddProcessorRingRequest, Orleans.GpuBridge.Backends.DotCompute, Version=0.1.0.0, Culture=neutral, PublicKeyToken=null]]' not found.
   at System.RuntimeType.CreateInstanceImpl(BindingFlags bindingAttr, Binder binder, Object[] args, CultureInfo culture)
   at DotCompute.Backends.CUDA.RingKernels.CudaMessageQueueBridgeFactory.CreateNamedQueueAsync(Type messageType, String queueName, MessageQueueOptions options, CancellationToken cancellationToken)
```

## Root Cause Analysis

### In `CudaMessageQueueBridgeFactory.cs` (lines 363-365):
```csharp
// Create instance - host-side queue doesn't need async initialization
var queue = Activator.CreateInstance(hostQueueType, options, logger)  // <-- Passes TWO args
    ?? throw new InvalidOperationException($"Failed to create host message queue for type {messageType.Name}");
```

### In `MessageQueue.cs` (lines 56-79):
```csharp
public MessageQueue(MessageQueueOptions options)  // <-- Only ONE parameter
{
    ArgumentNullException.ThrowIfNull(options);
    // ...
}
```

## Suggested Fix

### Option A: Update `MessageQueue<T>` to accept optional logger (Recommended)

Add a constructor overload that accepts an optional logger:

```csharp
public MessageQueue(MessageQueueOptions options, ILogger? logger = null)
{
    ArgumentNullException.ThrowIfNull(options);
    _logger = logger ?? NullLogger.Instance;
    // ... existing initialization
}
```

### Option B: Update `CudaMessageQueueBridgeFactory` to pass only options

Modify `CreateNamedQueueAsync` to only pass options:

```csharp
var queue = Activator.CreateInstance(hostQueueType, options)
    ?? throw new InvalidOperationException($"Failed to create host message queue for type {messageType.Name}");
```

## Additional Context

This bug was discovered while validating the GPU-native actor paradigm in Orleans.GpuBridge.Core. The CUDA message passing test cannot proceed past the "Creating bridged input queue" step due to this constructor mismatch.

### Test Command
```bash
cd tests/RingKernelValidation
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
dotnet run -- message-cuda
```

### Test Output Before Failure
```
[DIAG] LaunchAsync starting for kernel 'vectoradd_processor'
[DIAG] Step 1: Getting CUDA context...
[DIAG] Step 1: Got context 0x7487C0184C30
[DIAG] Step 2: Detecting message types...
[DIAG] Step 2: Detected Input=VectorAddProcessorRingRequest, Output=VectorAddProcessorRingRequest
[DIAG] Step 3a: Creating bridged input queue...

=== TEST FAILED ===
```

## Impact

- Blocks CUDA message passing tests
- Prevents validation of GPU-native actor paradigm
- Ring kernel message queue bridge cannot be created

## Priority

**High** - This blocks all CUDA ring kernel message passing functionality.
