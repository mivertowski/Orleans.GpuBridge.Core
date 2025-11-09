using System.Collections.Generic;
using ILGPU;

namespace Orleans.GpuBridge.Backends.ILGPU.Kernels;

/// <summary>
/// Kernel arguments wrapper
/// </summary>
public class KernelArguments
{
    private readonly List<object> _arguments = new();

    public void Add(object argument) => _arguments.Add(argument);

    public object[] ToArray() => _arguments.ToArray();

    public int GetWorkSize()
    {
        // Try to determine work size from first array argument
        foreach (var arg in _arguments)
        {
            if (arg is ArrayView<float> view)
                return (int)view.Length;
            if (arg is ArrayView2D<float, Stride2D.DenseX> view2d)
                return view2d.IntExtent.X * view2d.IntExtent.Y;
        }
        return 1024; // Default
    }

    public ArrayView<int> GetArrayView()
    {
        // Try to find an appropriate ArrayView in the arguments
        foreach (var arg in _arguments)
        {
            if (arg is ArrayView<int> intView)
                return intView;
        }

        // Return empty ArrayView as fallback
        return new ArrayView<int>();
    }
}
