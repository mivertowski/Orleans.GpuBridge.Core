using System.Diagnostics.CodeAnalysis;
using Orleans;

namespace Orleans.GpuBridge.Abstractions;

/// <summary>
/// Represents a unique kernel identifier
/// </summary>
[GenerateSerializer]
public readonly record struct KernelId([property: Id(0)] string Value)
{
    /// <summary>
    /// Returns the string representation of the kernel ID.
    /// </summary>
    /// <returns>The kernel ID value.</returns>
    public override string ToString() => Value;

    /// <summary>
    /// Parses a string into a <see cref="KernelId"/>.
    /// </summary>
    /// <param name="s">The string to parse.</param>
    /// <returns>A new <see cref="KernelId"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when the string is null or whitespace.</exception>
    public static KernelId Parse([NotNull] string s)
    {
        if (string.IsNullOrWhiteSpace(s))
            throw new ArgumentException("Kernel ID cannot be null or whitespace", nameof(s));
        return new(s);
    }

    /// <summary>
    /// Attempts to parse a string into a <see cref="KernelId"/>.
    /// </summary>
    /// <param name="s">The string to parse.</param>
    /// <param name="result">The parsed <see cref="KernelId"/> if successful.</param>
    /// <returns>True if parsing succeeded; otherwise, false.</returns>
    public static bool TryParse(string? s, out KernelId result)
    {
        if (string.IsNullOrWhiteSpace(s))
        {
            result = default;
            return false;
        }

        result = new KernelId(s);
        return true;
    }

    /// <summary>
    /// Implicitly converts a <see cref="KernelId"/> to its string value.
    /// </summary>
    /// <param name="id">The kernel ID to convert.</param>
    public static implicit operator string(KernelId id) => id.Value;
}
