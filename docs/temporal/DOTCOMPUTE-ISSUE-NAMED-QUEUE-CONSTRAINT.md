# DotCompute Issue: Named Message Queue Constraint Violation

**Date**: November 14, 2025
**DotCompute Version**: 0.5.1-alpha
**Severity**: Critical - Blocks message passing functionality
**Status**: Requires DotCompute team fix

---

## Issue Summary

When launching a ring kernel with `IRingKernelMessage`-based message types (classes), `CpuRingKernelRuntime.LaunchAsync` fails with a generic constraint violation because it attempts to call `CreateMessageQueueAsync<T>() where T : unmanaged` instead of `CreateNamedMessageQueueAsync<T>() where T : IRingKernelMessage`.

---

## Error Details

### Error Message
```
System.ArgumentException: GenericArguments[0], 'Orleans.GpuBridge.Backends.DotCompute.Temporal.VectorAddRequestMessage',
on 'System.Threading.Tasks.Task`1[DotCompute.Abstractions.RingKernels.IMessageQueue`1[T]] CreateMessageQueueAsync[T](Int32, System.Threading.CancellationToken)'
violates the constraint of type 'T'.

---> System.Security.VerificationException: Method DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.CreateMessageQueueAsync:
type argument 'Orleans.GpuBridge.Backends.DotCompute.Temporal.VectorAddRequestMessage' violates the constraint of type parameter 'T'.
```

### Stack Trace
```
at DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.CreateTypedMessageQueueAsync(Type messageType, Int32 capacity, CancellationToken cancellationToken)
at DotCompute.Backends.CPU.RingKernels.CpuRingKernelRuntime.<>c__DisplayClass6_0.<<LaunchAsync>b__0>d.MoveNext()
```

---

## Root Cause

The `CpuRingKernelRuntime.LaunchAsync` method (and likely `CudaRingKernelRuntime.LaunchAsync`) internally calls `CreateTypedMessageQueueAsync` which uses reflection to invoke `CreateMessageQueueAsync<T>()`.

However, in DotCompute v0.5.0-alpha, two different message queue creation APIs were introduced:

1. **Old API** (for unmanaged types - structs):
   ```csharp
   Task<IMessageQueue<T>> CreateMessageQueueAsync<T>(int capacity, CancellationToken cancellationToken)
       where T : unmanaged
   ```

2. **New API** (for IRingKernelMessage - classes):
   ```csharp
   Task<IMessageQueue<T>> CreateNamedMessageQueueAsync<T>(string queueName, int capacity, CancellationToken cancellationToken)
       where T : IRingKernelMessage
   ```

The ring kernel launch logic is still trying to use the old `CreateMessageQueueAsync<T>` method for all message types, including `IRingKernelMessage` types, which violates the `where T : unmanaged` constraint.

---

## Message Type Details

**VectorAddRequestMessage** (from VectorAddMessages.cs):
```csharp
public partial class VectorAddRequestMessage : IRingKernelMessage
{
    public Guid MessageId { get; set; }
    public byte Priority { get; set; }
    public int VectorALength { get; set; }
    public VectorOperation Operation { get; set; }
    public bool UseGpuMemory { get; set; }
    public ulong GpuBufferAHandleId { get; set; }
    public ulong GpuBufferBHandleId { get; set; }
    public ulong GpuBufferResultHandleId { get; set; }
    public float[] InlineDataA { get; set; } = Array.Empty<float>();
    public float[] InlineDataB { get; set; } = Array.Empty<float>();
}
```

This is a **class** implementing `IRingKernelMessage`, not an `unmanaged` struct.

---

## Expected Behavior

`CpuRingKernelRuntime.LaunchAsync` should detect whether the message type is:

1. **Unmanaged struct**: Call `CreateMessageQueueAsync<T>() where T : unmanaged`
2. **IRingKernelMessage class**: Call `CreateNamedMessageQueueAsync<T>() where T : IRingKernelMessage`

The detection can be done via:
```csharp
if (typeof(IRingKernelMessage).IsAssignableFrom(messageType))
{
    // Use CreateNamedMessageQueueAsync<T>
    var queueName = $"{kernelId}_{messagePropertyName}";
    await CreateNamedMessageQueueAsync<T>(queueName, capacity, cancellationToken);
}
else if (IsUnmanagedType(messageType))
{
    // Use CreateMessageQueueAsync<T>
    await CreateMessageQueueAsync<T>(capacity, cancellationToken);
}
```

---

## Affected Code

**File**: `DotCompute/DotCompute/src/Backends/DotCompute.Backends.CPU/RingKernels/CpuRingKernelRuntime.cs`
**Method**: `LaunchAsync` → `CreateTypedMessageQueueAsync`

**Expected Locations**:
- `CpuRingKernelRuntime.LaunchAsync`
- `CudaRingKernelRuntime.LaunchAsync`
- Any other runtime that creates message queues during kernel launch

---

## Suggested Fix

Update the `CreateTypedMessageQueueAsync` method (or wherever reflection-based queue creation happens) to:

```csharp
private async Task CreateTypedMessageQueueAsync(
    Type messageType,
    string kernelId,
    string queueName,
    int capacity,
    CancellationToken cancellationToken)
{
    if (typeof(IRingKernelMessage).IsAssignableFrom(messageType))
    {
        // New API: Named message queue for IRingKernelMessage types
        var method = GetType()
            .GetMethod(nameof(CreateNamedMessageQueueAsync))!
            .MakeGenericMethod(messageType);

        var fullQueueName = $"{kernelId}_{queueName}";
        await (Task)method.Invoke(this, new object[] { fullQueueName, capacity, cancellationToken })!;
    }
    else if (IsUnmanagedType(messageType))
    {
        // Old API: Unnamed message queue for unmanaged types
        var method = GetType()
            .GetMethod(nameof(CreateMessageQueueAsync))!
            .MakeGenericMethod(messageType);

        await (Task)method.Invoke(this, new object[] { capacity, cancellationToken })!;
    }
    else
    {
        throw new ArgumentException($"Message type {messageType.FullName} must be either unmanaged or implement IRingKernelMessage");
    }
}

private static bool IsUnmanagedType(Type type)
{
    if (!type.IsValueType)
        return false;

    if (type.IsPrimitive || type.IsPointer || type.IsEnum)
        return true;

    return type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
        .All(f => IsUnmanagedType(f.FieldType));
}
```

---

## Workaround

None available. This blocks all ring kernels using `IRingKernelMessage`-based message types.

---

## Test Case

**File**: `Orleans.GpuBridge.Core/tests/RingKernelValidation/MessagePassingTest.cs`
**Command**: `dotnet run --project tests/RingKernelValidation/RingKernelValidation.csproj -- message`

**Expected Result**: Message passing test completes successfully
**Actual Result**: Fails with constraint violation during kernel launch (line 46 of MessagePassingTest.cs)

---

## Impact

- **Orleans.GpuBridge.Core**: Cannot use ring kernels with structured message types
- **All DotCompute users**: Cannot use named message queues with `IRingKernelMessage` types
- **Backward compatibility**: Existing code using unmanaged structs should continue to work

---

## Related Files

1. `Orleans.GpuBridge.Core/src/Orleans.GpuBridge.Backends.DotCompute/Temporal/VectorAddMessages.cs` - Message definitions
2. `Orleans.GpuBridge.Core/tests/RingKernelValidation/MessagePassingTest.cs` - Failing test
3. `DotCompute/DotCompute/src/Backends/DotCompute.Backends.CPU/RingKernels/CpuRingKernelRuntime.cs` - Needs fix
4. `DotCompute/DotCompute/src/Backends/DotCompute.Backends.CUDA/RingKernels/CudaRingKernelRuntime.cs` - Needs fix

---

## Priority

**Critical** - This blocks the entire Phase 5 Ring Kernel Integration milestone. Without this fix, we cannot:
- Test message passing functionality
- Validate 100-500ns latency targets
- Validate 2M+ messages/s throughput targets
- Complete GPU-native actor implementation

---

**Next Action**: DotCompute team needs to implement the suggested fix in v0.5.2-alpha or provide an alternative solution.
