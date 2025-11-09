using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace Orleans.GpuBridge.Performance;

/// <summary>
/// Unmanaged memory manager for NUMA-allocated memory
/// </summary>
internal unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
{
    private readonly T* _pointer;
    private readonly int _length;
    private readonly IntPtr _originalPointer;

    public UnmanagedMemoryManager(T* pointer, int length, IntPtr originalPointer)
    {
        _pointer = pointer;
        _length = length;
        _originalPointer = originalPointer;
    }

    protected override void Dispose(bool disposing)
    {
        if (_originalPointer != IntPtr.Zero)
        {
            VirtualFree(_originalPointer, 0, FreeType.Release);
        }
    }

    public override Span<T> GetSpan() => new(_pointer, _length);

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > (uint)_length)
            throw new ArgumentOutOfRangeException(nameof(elementIndex));

        return new MemoryHandle(_pointer + elementIndex);
    }

    public override void Unpin() { }

    [DllImport("kernel32.dll")]
    private static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, FreeType dwFreeType);

    private enum FreeType : uint
    {
        Release = 0x8000
    }
}
