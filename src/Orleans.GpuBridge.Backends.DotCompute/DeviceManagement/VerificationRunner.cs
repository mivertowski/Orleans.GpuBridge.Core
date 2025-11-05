// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Extensions.Logging;

namespace Orleans.GpuBridge.Backends.DotCompute.DeviceManagement;

/// <summary>
/// Simple runner for API verification
/// </summary>
internal static class VerificationRunner
{
    /// <summary>
    /// Main entry point for verification
    /// </summary>
    public static async Task<int> Main(string[] args)
    {
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information);
        });

        var logger = loggerFactory.CreateLogger("DotComputeVerification");

        try
        {
            var result = await DotComputeApiVerification.VerifyApisAsync(logger);

            Console.WriteLine();
            Console.WriteLine(result.GetSummary());
            Console.WriteLine();

            return result.Success ? 0 : 1;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Verification runner failed");
            return 1;
        }
    }
}
