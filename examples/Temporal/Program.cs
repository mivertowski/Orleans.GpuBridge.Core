using System;

namespace Orleans.GpuBridge.Examples.Temporal;

/// <summary>
/// Main program to run all Phase 1 temporal correctness examples.
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║  Orleans.GpuBridge.Core - Temporal Correctness Examples      ║");
        Console.WriteLine("║  Phase 1: HLC, Message Passing, and Pattern Detection        ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        if (args.Length > 0 && int.TryParse(args[0], out int exampleNumber))
        {
            RunExample(exampleNumber);
        }
        else
        {
            RunAllExamples();
        }

        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║  Examples Complete!                                           ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }

    static void RunAllExamples()
    {
        Console.WriteLine("Running all examples...\n");

        // Example 1: Basic HLC
        RunExample(1);
        WaitForUser();

        // Example 2: Message Passing
        RunExample(2);
        WaitForUser();

        // Example 3: Financial Transactions
        RunExample(3);
        WaitForUser();
    }

    static void RunExample(int number)
    {
        try
        {
            switch (number)
            {
                case 1:
                    Console.WriteLine("\n" + new string('═', 65));
                    BasicHLCExample.Run();
                    BasicHLCExample.DemonstrateDriftMeasurement();
                    BasicHLCExample.DemonstrateClockSources();
                    break;

                case 2:
                    Console.WriteLine("\n" + new string('═', 65));
                    MessagePassingExample.Run();
                    MessagePassingExample.DemonstrateDiamondDependency();
                    MessagePassingExample.DemonstratePriorityProcessing();
                    MessagePassingExample.DemonstrateDeadlineEviction();
                    break;

                case 3:
                    Console.WriteLine("\n" + new string('═', 65));
                    FinancialTransactionExample.Run();
                    FinancialTransactionExample.DemonstrateCircularTransactionDetection();
                    break;

                default:
                    Console.WriteLine($"Unknown example number: {number}");
                    Console.WriteLine("Valid examples: 1-3");
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ Error running example {number}:");
            Console.WriteLine($"  {ex.Message}");
            Console.WriteLine($"\nStack trace:");
            Console.WriteLine(ex.StackTrace);
        }
    }

    static void WaitForUser()
    {
        Console.WriteLine("\n" + new string('─', 65));
        Console.WriteLine("Press any key to continue to next example...");
        Console.ReadKey();
    }
}
