// Orleans.GpuBridge - GPU-native distributed computing for Microsoft Orleans
// Copyright (c) 2025 Michael Ivertowski

using System;
using Microsoft.CodeAnalysis;

namespace Orleans.GpuBridge.Generators.Analysis;

/// <summary>
/// Calculates message sizes and validates blittability of types.
/// </summary>
public static class MessageSizeCalculator
{
    /// <summary>
    /// Determines if a type is blittable (can be directly copied to GPU memory).
    /// </summary>
    /// <param name="type">The type to check.</param>
    /// <returns>True if the type is blittable.</returns>
    public static bool IsBlittable(ITypeSymbol type)
    {
        // Primitive numeric types are always blittable
        if (IsPrimitiveBlittable(type))
        {
            return true;
        }

        // Enums are blittable
        if (type.TypeKind == TypeKind.Enum)
        {
            return true;
        }

        // Arrays are not blittable (reference types)
        if (type.TypeKind == TypeKind.Array)
        {
            return false;
        }

        // Strings are not blittable
        if (type.SpecialType == SpecialType.System_String)
        {
            return false;
        }

        // Object and dynamic are not blittable
        if (type.SpecialType == SpecialType.System_Object)
        {
            return false;
        }

        // Check if it's a struct
        if (type.TypeKind == TypeKind.Struct)
        {
            return IsBlittableStruct(type);
        }

        // Reference types are not blittable
        if (type.IsReferenceType)
        {
            return false;
        }

        return false;
    }

    /// <summary>
    /// Gets the size of a type in bytes.
    /// </summary>
    /// <param name="type">The type to measure.</param>
    /// <param name="paddingBytes">Output: bytes of padding due to alignment.</param>
    /// <returns>Size in bytes, or 0 for variable-size types.</returns>
    public static int GetTypeSize(ITypeSymbol type, out int paddingBytes)
    {
        paddingBytes = 0;

        // Handle primitive types
        var primitiveSize = GetPrimitiveSize(type);
        if (primitiveSize > 0)
        {
            return primitiveSize;
        }

        // Handle enums
        if (type.TypeKind == TypeKind.Enum)
        {
            var enumType = (INamedTypeSymbol)type;
            return GetTypeSize(enumType.EnumUnderlyingType!, out paddingBytes);
        }

        // Arrays have variable size
        if (type.TypeKind == TypeKind.Array)
        {
            return 0; // Variable size
        }

        // Handle structs
        if (type.TypeKind == TypeKind.Struct)
        {
            return CalculateStructSize(type, out paddingBytes);
        }

        // Handle well-known types
        var wellKnownSize = GetWellKnownTypeSize(type);
        if (wellKnownSize > 0)
        {
            return wellKnownSize;
        }

        return 0; // Unknown size
    }

    private static bool IsPrimitiveBlittable(ITypeSymbol type)
    {
        return type.SpecialType switch
        {
            SpecialType.System_Boolean => true,
            SpecialType.System_Byte => true,
            SpecialType.System_SByte => true,
            SpecialType.System_Int16 => true,
            SpecialType.System_UInt16 => true,
            SpecialType.System_Int32 => true,
            SpecialType.System_UInt32 => true,
            SpecialType.System_Int64 => true,
            SpecialType.System_UInt64 => true,
            SpecialType.System_Single => true,
            SpecialType.System_Double => true,
            SpecialType.System_IntPtr => true,
            SpecialType.System_UIntPtr => true,
            SpecialType.System_Char => true,
            _ => false
        };
    }

    private static int GetPrimitiveSize(ITypeSymbol type)
    {
        return type.SpecialType switch
        {
            SpecialType.System_Boolean => 1,
            SpecialType.System_Byte => 1,
            SpecialType.System_SByte => 1,
            SpecialType.System_Int16 => 2,
            SpecialType.System_UInt16 => 2,
            SpecialType.System_Char => 2,
            SpecialType.System_Int32 => 4,
            SpecialType.System_UInt32 => 4,
            SpecialType.System_Single => 4,
            SpecialType.System_Int64 => 8,
            SpecialType.System_UInt64 => 8,
            SpecialType.System_Double => 8,
            SpecialType.System_IntPtr => 8, // Assume 64-bit
            SpecialType.System_UIntPtr => 8, // Assume 64-bit
            _ => 0
        };
    }

    private static int GetWellKnownTypeSize(ITypeSymbol type)
    {
        var fullName = type.ToDisplayString();

        return fullName switch
        {
            // System types
            "System.Guid" => 16,
            "System.DateTime" => 8,
            "System.DateTimeOffset" => 16,
            "System.TimeSpan" => 8,
            "System.Decimal" => 16,
            "System.Half" => 2,
            "System.Int128" => 16,
            "System.UInt128" => 16,

            // Common struct types
            "System.Numerics.Vector2" => 8,
            "System.Numerics.Vector3" => 12,
            "System.Numerics.Vector4" => 16,
            "System.Numerics.Matrix3x2" => 24,
            "System.Numerics.Matrix4x4" => 64,
            "System.Numerics.Quaternion" => 16,
            "System.Numerics.Plane" => 16,

            // Orleans.GpuBridge temporal types
            "Orleans.GpuBridge.Abstractions.Temporal.HybridTimestamp" => 16,

            _ => 0
        };
    }

    private static bool IsBlittableStruct(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol namedType)
        {
            return false;
        }

        // Check each field
        foreach (var member in namedType.GetMembers())
        {
            if (member is IFieldSymbol field && !field.IsStatic)
            {
                if (!IsBlittable(field.Type))
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static int CalculateStructSize(ITypeSymbol type, out int paddingBytes)
    {
        paddingBytes = 0;

        if (type is not INamedTypeSymbol namedType)
        {
            return 0;
        }

        // Check for well-known types first
        var wellKnownSize = GetWellKnownTypeSize(type);
        if (wellKnownSize > 0)
        {
            return wellKnownSize;
        }

        // Calculate size based on fields
        int currentOffset = 0;
        int maxAlignment = 1;

        foreach (var member in namedType.GetMembers())
        {
            if (member is IFieldSymbol field && !field.IsStatic)
            {
                int fieldSize = GetTypeSize(field.Type, out _);
                if (fieldSize == 0)
                {
                    return 0; // Unknown size if any field has unknown size
                }

                // Calculate alignment (assume natural alignment up to 8 bytes)
                int alignment = Math.Min(fieldSize, 8);
                maxAlignment = Math.Max(maxAlignment, alignment);

                // Add padding for alignment
                int padding = (alignment - (currentOffset % alignment)) % alignment;
                paddingBytes += padding;
                currentOffset += padding;

                // Add field size
                currentOffset += fieldSize;
            }
        }

        // Final padding for struct alignment
        int finalPadding = (maxAlignment - (currentOffset % maxAlignment)) % maxAlignment;
        paddingBytes += finalPadding;
        currentOffset += finalPadding;

        return currentOffset;
    }

    /// <summary>
    /// Checks if a struct has excessive padding that could be optimized.
    /// </summary>
    /// <param name="type">The type to check.</param>
    /// <param name="totalSize">The total size of the struct.</param>
    /// <param name="paddingBytes">The number of padding bytes.</param>
    /// <returns>True if padding is more than 25% of the struct size.</returns>
    public static bool HasExcessivePadding(ITypeSymbol type, out int totalSize, out int paddingBytes)
    {
        totalSize = GetTypeSize(type, out paddingBytes);

        if (totalSize == 0)
        {
            return false;
        }

        // Consider excessive if padding is more than 25% of size
        return paddingBytes > totalSize / 4;
    }

    /// <summary>
    /// Gets the alignment requirement for a type.
    /// </summary>
    /// <param name="type">The type to check.</param>
    /// <returns>Alignment in bytes.</returns>
    public static int GetAlignment(ITypeSymbol type)
    {
        var size = GetTypeSize(type, out _);
        return size > 0 ? Math.Min(size, 8) : 8;
    }
}
