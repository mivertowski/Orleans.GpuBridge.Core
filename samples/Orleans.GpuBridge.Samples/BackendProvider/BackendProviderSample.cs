using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Orleans.GpuBridge.Abstractions.Providers;
using Orleans.GpuBridge.Abstractions.Enums;
using Orleans.GpuBridge.Runtime.Providers;
using Spectre.Console;

namespace Orleans.GpuBridge.Samples.BackendProvider;

/// <summary>
/// Sample demonstrating the backend provider selection and capabilities
/// </summary>
public class BackendProviderSample
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<BackendProviderSample> _logger;

    public BackendProviderSample(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _logger = serviceProvider.GetRequiredService<ILogger<BackendProviderSample>>();
    }

    public async Task RunAsync()
    {
        var table = new Table();
        table.AddColumn("Task");
        table.AddColumn("Status");
        table.AddColumn("Details");

        AnsiConsole.Live(table)
            .Start(ctx =>
            {
                // Initialize registry
                table.AddRow("Initializing Backend Registry", "[yellow]In Progress[/]", "...");
                ctx.Refresh();

                var registry = _serviceProvider.GetRequiredService<IGpuBackendRegistry>();
                registry.InitializeAsync().Wait();

                table.UpdateCell(table.Rows.Count - 1, 1, "[green]Complete[/]");
                ctx.Refresh();

                // Discover available providers
                table.AddRow("Discovering Providers", "[yellow]In Progress[/]", "...");
                ctx.Refresh();

                var availableProviders = registry.GetRegisteredProvidersAsync().Result;
                var providerCount = availableProviders.Count();

                table.UpdateCell(table.Rows.Count - 1, 1, "[green]Complete[/]");
                table.UpdateCell(table.Rows.Count - 1, 2, $"Found {providerCount} providers");
                ctx.Refresh();

                // Test provider selection
                table.AddRow("Testing Provider Selection", "[yellow]In Progress[/]", "...");
                ctx.Refresh();

                var selector = _serviceProvider.GetRequiredService<IGpuBridgeProviderSelector>();
                TestProviderSelection(selector).Wait();

                table.UpdateCell(table.Rows.Count - 1, 1, "[green]Complete[/]");
                table.UpdateCell(table.Rows.Count - 1, 2, "All tests passed");
                ctx.Refresh();
            });

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold green]Backend Provider Sample completed successfully![/]");
    }

    public async Task RunInteractiveAsync()
    {
        AnsiConsole.MarkupLine("[bold yellow]Backend Provider Interactive Sample[/]");
        AnsiConsole.WriteLine();

        var registry = _serviceProvider.GetRequiredService<IGpuBackendRegistry>();
        await registry.InitializeAsync();

        var selector = _serviceProvider.GetRequiredService<IGpuBridgeProviderSelector>();

        while (true)
        {
            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[yellow]What would you like to do?[/]")
                    .AddChoices(new[]
                    {
                        "List Available Providers",
                        "Test Provider Selection",
                        "Compare Provider Performance",
                        "View Provider Capabilities",
                        "Health Check All Providers",
                        "Simulate Provider Switching",
                        "Exit"
                    }));

            switch (choice)
            {
                case "List Available Providers":
                    await ListAvailableProvidersAsync(registry);
                    break;
                case "Test Provider Selection":
                    await TestProviderSelectionInteractiveAsync(selector);
                    break;
                case "Compare Provider Performance":
                    await CompareProviderPerformanceAsync(selector);
                    break;
                case "View Provider Capabilities":
                    await ViewProviderCapabilitiesAsync(registry);
                    break;
                case "Health Check All Providers":
                    await HealthCheckAllProvidersAsync(registry);
                    break;
                case "Simulate Provider Switching":
                    await SimulateProviderSwitchingAsync(selector);
                    break;
                case "Exit":
                    return;
            }

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[dim]Press any key to continue...[/]");
            Console.ReadKey(true);
            AnsiConsole.Clear();
        }
    }

    private async Task TestProviderSelection(IGpuBridgeProviderSelector selector)
    {
        // Test different selection criteria
        var testCases = new[]
        {
            new TestGpuExecutionRequirements { PreferGpu = true },
            new TestGpuExecutionRequirements { PreferGpu = false },
            new TestGpuExecutionRequirements 
            { 
                PreferGpu = true,
                RequiredCapabilities = new BackendCapabilities
                {
                    SupportedBackends = new[] { GpuBackend.Cuda }
                }
            }
        };

        foreach (var requirements in testCases)
        {
            var provider = await selector.SelectProviderAsync(requirements);
            _logger.LogInformation("Selected provider: {ProviderId} for requirements: GPU={PreferGpu}", 
                provider?.ProviderId ?? "None", requirements.PreferGpu);
        }
    }

    private async Task ListAvailableProvidersAsync(IGpuBackendRegistry registry)
    {
        AnsiConsole.MarkupLine("[bold]Available Backend Providers:[/]");
        
        var providers = await registry.GetRegisteredProvidersAsync();
        var table = new Table();
        table.AddColumn("Provider ID");
        table.AddColumn("Available");
        table.AddColumn("Supported Backends");
        table.AddColumn("JIT Support");
        table.AddColumn("Unified Memory");

        foreach (var provider in providers)
        {
            var isAvailable = provider.IsAvailable() ? "[green]Yes[/]" : "[red]No[/]";
            var backends = string.Join(", ", provider.Capabilities.SupportedBackends);
            var jitSupport = provider.Capabilities.SupportsJitCompilation ? "[green]Yes[/]" : "[red]No[/]";
            var unifiedMemory = provider.Capabilities.SupportsUnifiedMemory ? "[green]Yes[/]" : "[red]No[/]";

            table.AddRow(provider.ProviderId, isAvailable, backends, jitSupport, unifiedMemory);
        }

        AnsiConsole.Write(table);
    }

    private async Task TestProviderSelectionInteractiveAsync(IGpuBridgeProviderSelector selector)
    {
        AnsiConsole.MarkupLine("[bold]Testing Provider Selection[/]");
        
        var preferGpu = AnsiConsole.Confirm("Prefer GPU over CPU?", true);
        var allowFallback = AnsiConsole.Confirm("Allow CPU fallback?", true);
        
        var requirements = new TestGpuExecutionRequirements 
        { 
            PreferGpu = preferGpu,
            AllowCpuFallback = allowFallback
        };

        AnsiConsole.MarkupLine($"[yellow]Selection criteria:[/]");
        AnsiConsole.MarkupLine($"  Prefer GPU: {preferGpu}");
        AnsiConsole.MarkupLine($"  Allow fallback: {allowFallback}");

        var provider = await selector.SelectProviderAsync(requirements);
        
        if (provider != null)
        {
            AnsiConsole.MarkupLine($"[green]Selected provider: {provider.ProviderId}[/]");
            
            var devices = provider.GetDeviceManager().GetDevices();
            AnsiConsole.MarkupLine($"  Available devices: {devices.Count()}");
            
            foreach (var device in devices.Take(3))
            {
                AnsiConsole.MarkupLine($"    - {device.Name} ({device.Type})");
            }
        }
        else
        {
            AnsiConsole.MarkupLine("[red]No suitable provider found[/]");
        }
    }

    private async Task CompareProviderPerformanceAsync(IGpuBridgeProviderSelector selector)
    {
        AnsiConsole.MarkupLine("[bold]Comparing Provider Performance[/]");
        
        var providers = await selector.GetAvailableProvidersAsync();
        var results = new List<(string ProviderId, TimeSpan InitTime, bool Success)>();

        foreach (var provider in providers)
        {
            AnsiConsole.MarkupLine($"Testing {provider.ProviderId}...");
            
            var stopwatch = Stopwatch.StartNew();
            var success = false;
            
            try
            {
                // Simulate some initialization work
                var deviceManager = provider.GetDeviceManager();
                var devices = deviceManager.GetDevices();
                var defaultDevice = deviceManager.GetDefaultDevice();
                
                await Task.Delay(100); // Simulate initialization time
                success = true;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]  Error: {ex.Message}[/]");
            }
            
            stopwatch.Stop();
            results.Add((provider.ProviderId, stopwatch.Elapsed, success));
        }

        // Display results
        var table = new Table();
        table.AddColumn("Provider");
        table.AddColumn("Initialization Time");
        table.AddColumn("Status");

        foreach (var (providerId, initTime, success) in results.OrderBy(r => r.InitTime))
        {
            var status = success ? "[green]Success[/]" : "[red]Failed[/]";
            table.AddRow(providerId, $"{initTime.TotalMilliseconds:F1}ms", status);
        }

        AnsiConsole.Write(table);
    }

    private async Task ViewProviderCapabilitiesAsync(IGpuBackendRegistry registry)
    {
        AnsiConsole.MarkupLine("[bold]Provider Capabilities Comparison[/]");
        
        var providers = await registry.GetRegisteredProvidersAsync();
        
        foreach (var provider in providers)
        {
            var panel = new Panel($"""
                [bold]Supported Backends:[/] {string.Join(", ", provider.Capabilities.SupportedBackends)}
                [bold]Max Concurrent Devices:[/] {provider.Capabilities.MaxConcurrentDevices}
                [bold]JIT Compilation:[/] {(provider.Capabilities.SupportsJitCompilation ? "✓" : "✗")}
                [bold]AOT Compilation:[/] {(provider.Capabilities.SupportsAotCompilation ? "✓" : "✗")}
                [bold]Unified Memory:[/] {(provider.Capabilities.SupportsUnifiedMemory ? "✓" : "✗")}
                [bold]Atomic Operations:[/] {(provider.Capabilities.SupportsAtomicOperations ? "✓" : "✗")}
                [bold]Tensor Operations:[/] {(provider.Capabilities.SupportsTensorOperations ? "✓" : "✗")}
                [bold]CPU Debugging:[/] {(provider.Capabilities.SupportsCpuDebugging ? "✓" : "✗")}
                [bold]Profiling:[/] {(provider.Capabilities.SupportsProfiling ? "✓" : "✗")}
                [bold]Supported Languages:[/] {string.Join(", ", provider.Capabilities.SupportedKernelLanguages)}
                [bold]Supported Platforms:[/] {string.Join(", ", provider.Capabilities.SupportedPlatforms)}
                """)
            {
                Header = new PanelHeader($" {provider.ProviderId} "),
                Border = BoxBorder.Rounded
            };

            AnsiConsole.Write(panel);
            AnsiConsole.WriteLine();
        }
    }

    private async Task HealthCheckAllProvidersAsync(IGpuBackendRegistry registry)
    {
        AnsiConsole.MarkupLine("[bold]Health Checking All Providers[/]");
        
        var providers = await registry.GetRegisteredProvidersAsync();
        var table = new Table();
        table.AddColumn("Provider");
        table.AddColumn("Status");
        table.AddColumn("Message");
        table.AddColumn("Response Time");

        foreach (var provider in providers)
        {
            var stopwatch = Stopwatch.StartNew();
            var health = await provider.CheckHealthAsync();
            stopwatch.Stop();

            var status = health.IsHealthy ? "[green]Healthy[/]" : "[red]Unhealthy[/]";
            var message = health.Message ?? "OK";
            var responseTime = $"{stopwatch.ElapsedMilliseconds}ms";

            table.AddRow(provider.ProviderId, status, message, responseTime);
        }

        AnsiConsole.Write(table);
    }

    private async Task SimulateProviderSwitchingAsync(IGpuBridgeProviderSelector selector)
    {
        AnsiConsole.MarkupLine("[bold]Simulating Provider Switching Scenarios[/]");
        
        var scenarios = new[]
        {
            ("High Performance Required", new TestGpuExecutionRequirements 
            { 
                PreferGpu = true,
                RequiredCapabilities = new BackendCapabilities 
                { 
                    SupportedBackends = new[] { GpuBackend.Cuda },
                    SupportsTensorOperations = true
                }
            }),
            ("Memory Constrained", new TestGpuExecutionRequirements 
            { 
                PreferGpu = true,
                MinimumMemoryBytes = 128 * 1024 * 1024 // 128 MB
            }),
            ("Debugging Mode", new TestGpuExecutionRequirements 
            { 
                PreferGpu = false,
                RequiredCapabilities = new BackendCapabilities 
                { 
                    SupportsCpuDebugging = true
                }
            }),
            ("Cross Platform", new TestGpuExecutionRequirements 
            { 
                PreferGpu = true,
                AllowCpuFallback = true
            })
        };

        var table = new Table();
        table.AddColumn("Scenario");
        table.AddColumn("Selected Provider");
        table.AddColumn("Backend Type");
        table.AddColumn("Reason");

        foreach (var (scenarioName, requirements) in scenarios)
        {
            var provider = await selector.SelectProviderAsync(requirements);
            
            if (provider != null)
            {
                var backendType = provider.Capabilities.SupportedBackends.FirstOrDefault()?.ToString() ?? "Unknown";
                var reason = DetermineSelectionReason(provider, requirements);
                
                table.AddRow(
                    scenarioName, 
                    provider.ProviderId, 
                    backendType, 
                    reason
                );
            }
            else
            {
                table.AddRow(
                    scenarioName, 
                    "[red]None[/]", 
                    "[red]N/A[/]", 
                    "No suitable provider found"
                );
            }
        }

        AnsiConsole.Write(table);
    }

    private string DetermineSelectionReason(IGpuBackendProvider provider, TestGpuExecutionRequirements requirements)
    {
        if (!requirements.PreferGpu && provider.Capabilities.SupportedBackends.Contains(GpuBackend.Cpu))
            return "CPU preference";
        
        if (requirements.PreferGpu && provider.Capabilities.SupportedBackends.Any(b => b != GpuBackend.Cpu))
            return "GPU preference";
        
        if (requirements.AllowCpuFallback && provider.Capabilities.SupportedBackends.Contains(GpuBackend.Cpu))
            return "CPU fallback";
        
        return "Best match";
    }
}

/// <summary>
/// Test implementation for GPU execution requirements
/// </summary>
internal class TestGpuExecutionRequirements
{
    public bool PreferGpu { get; set; }
    public bool AllowCpuFallback { get; set; } = true;
    public long MinimumMemoryBytes { get; set; }
    public BackendCapabilities? RequiredCapabilities { get; set; }
    public List<GpuBackend> PreferredBackends { get; set; } = new();
    public int MaxDevices { get; set; } = 1;
}