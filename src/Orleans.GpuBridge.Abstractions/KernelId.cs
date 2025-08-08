namespace Orleans.GpuBridge.Abstractions;
public readonly record struct KernelId(string Value)
{
    public override string ToString() => Value;
    public static KernelId Parse(string s) => new(s);
}
