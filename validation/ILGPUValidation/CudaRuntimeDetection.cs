using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Orleans.GpuBridge.Examples
{
    /// <summary>
    /// Native CUDA runtime detection to identify why ILGPU doesn't see GPU
    /// </summary>
    public static class CudaRuntimeDetection
    {
        // CUDA Runtime API P/Invoke declarations
        [DllImport("cudart", EntryPoint = "cudaGetDeviceCount")]
        private static extern int CudaGetDeviceCount(out int count);

        [DllImport("cudart", EntryPoint = "cudaGetDeviceProperties")]
        private static extern int CudaGetDeviceProperties(out CudaDeviceProperties prop, int device);

        [DllImport("cudart", EntryPoint = "cudaRuntimeGetVersion")]
        private static extern int CudaRuntimeGetVersion(out int runtimeVersion);

        [DllImport("cudart", EntryPoint = "cudaDriverGetVersion")]
        private static extern int CudaDriverGetVersion(out int driverVersion);

        [DllImport("cudart", EntryPoint = "cudaSetDevice")]
        private static extern int CudaSetDevice(int device);

        [StructLayout(LayoutKind.Sequential)]
        private struct CudaDeviceProperties
        {
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
            public byte[] name;
            public IntPtr totalGlobalMem;
            public IntPtr sharedMemPerBlock;
            public int regsPerBlock;
            public int warpSize;
            public IntPtr memPitch;
            public int maxThreadsPerBlock;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] maxThreadsDim;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] maxGridSize;
            public int clockRate;
            public IntPtr totalConstMem;
            public int major;
            public int minor;
            public IntPtr textureAlignment;
            public int deviceOverlap;
            public int multiProcessorCount;
            public int kernelExecTimeoutEnabled;
            public int integrated;
            public int canMapHostMemory;
            public int computeMode;
            public int maxTexture1D;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
            public int[] maxTexture2D;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
            public int[] maxTexture3D;
        }

        public static void RunCudaRuntimeDetection()
        {
            Console.WriteLine("🔍 Native CUDA Runtime Detection");
            Console.WriteLine("=================================");

            // Test 1: Check if CUDA runtime library can be loaded
            Console.WriteLine("Step 1: Testing CUDA Runtime Library Loading...");
            try
            {
                int deviceCount = 0;
                int result = CudaGetDeviceCount(out deviceCount);
                
                if (result == 0) // cudaSuccess
                {
                    Console.WriteLine($"✅ CUDA Runtime loaded successfully");
                    Console.WriteLine($"✅ Found {deviceCount} CUDA device(s) via native API");
                }
                else
                {
                    Console.WriteLine($"❌ CUDA Runtime returned error code: {result}");
                    PrintCudaErrorCode(result);
                    return;
                }

                if (deviceCount == 0)
                {
                    Console.WriteLine("❌ No CUDA devices found via native API");
                    Console.WriteLine("   This indicates either:");
                    Console.WriteLine("   - No NVIDIA GPU present");
                    Console.WriteLine("   - GPU driver not installed");
                    Console.WriteLine("   - WSL2 GPU support not enabled");
                    return;
                }

                // Test 2: Get CUDA version information
                Console.WriteLine("\nStep 2: CUDA Version Information...");
                int runtimeVersion = 0, driverVersion = 0;
                
                result = CudaRuntimeGetVersion(out runtimeVersion);
                if (result == 0)
                {
                    Console.WriteLine($"✅ CUDA Runtime Version: {runtimeVersion / 1000}.{(runtimeVersion % 100) / 10}");
                }
                
                result = CudaDriverGetVersion(out driverVersion);
                if (result == 0)
                {
                    Console.WriteLine($"✅ CUDA Driver Version: {driverVersion / 1000}.{(driverVersion % 100) / 10}");
                }

                // Test 3: Query device properties
                Console.WriteLine("\nStep 3: Device Properties...");
                for (int i = 0; i < deviceCount; i++)
                {
                    CudaDeviceProperties props;
                    result = CudaGetDeviceProperties(out props, i);
                    
                    if (result == 0)
                    {
                        string deviceName = Encoding.ASCII.GetString(props.name).TrimEnd('\0');
                        Console.WriteLine($"✅ Device {i}: {deviceName}");
                        Console.WriteLine($"   Compute Capability: {props.major}.{props.minor}");
                        Console.WriteLine($"   Global Memory: {props.totalGlobalMem.ToInt64() / (1024 * 1024)} MB");
                        Console.WriteLine($"   Multiprocessors: {props.multiProcessorCount}");
                        Console.WriteLine($"   Warp Size: {props.warpSize}");
                        Console.WriteLine($"   Max Threads/Block: {props.maxThreadsPerBlock}");
                    }
                    else
                    {
                        Console.WriteLine($"❌ Failed to get properties for device {i}: error {result}");
                    }
                }

                // Test 4: Try to set device
                Console.WriteLine("\nStep 4: Device Initialization Test...");
                result = CudaSetDevice(0);
                if (result == 0)
                {
                    Console.WriteLine("✅ Successfully set CUDA device 0");
                    Console.WriteLine("✅ Native CUDA runtime is fully functional");
                }
                else
                {
                    Console.WriteLine($"❌ Failed to set CUDA device 0: error {result}");
                }

            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"❌ CUDA Runtime Library not found: {ex.Message}");
                Console.WriteLine("   Possible causes:");
                Console.WriteLine("   - CUDA not installed");
                Console.WriteLine("   - CUDA lib64 not in LD_LIBRARY_PATH");
                Console.WriteLine("   - WSL2 with incorrect CUDA installation");
                
                Console.WriteLine("\nTrying to locate CUDA libraries...");
                var cudaLibPaths = new[] { 
                    "/usr/local/cuda/lib64/libcudart.so",
                    "/usr/local/cuda-13.0/lib64/libcudart.so",
                    "/usr/local/cuda-12.6/lib64/libcudart.so"
                };

                foreach (var path in cudaLibPaths)
                {
                    if (System.IO.File.Exists(path))
                    {
                        Console.WriteLine($"✅ Found CUDA library: {path}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Unexpected error: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            }
        }

        private static void PrintCudaErrorCode(int errorCode)
        {
            string errorMessage = errorCode switch
            {
                1 => "cudaErrorMissingConfiguration",
                2 => "cudaErrorMemoryAllocation", 
                3 => "cudaErrorInitializationError",
                4 => "cudaErrorLaunchFailure",
                5 => "cudaErrorPriorLaunchFailure",
                6 => "cudaErrorLaunchTimeout",
                7 => "cudaErrorLaunchOutOfResources",
                8 => "cudaErrorInvalidDeviceFunction",
                9 => "cudaErrorInvalidConfiguration",
                10 => "cudaErrorInvalidDevice",
                11 => "cudaErrorInvalidValue",
                12 => "cudaErrorInvalidPitchValue",
                13 => "cudaErrorInvalidSymbol",
                14 => "cudaErrorMapBufferObjectFailed",
                15 => "cudaErrorUnmapBufferObjectFailed",
                16 => "cudaErrorInvalidHostPointer",
                17 => "cudaErrorInvalidDevicePointer",
                18 => "cudaErrorInvalidTexture",
                19 => "cudaErrorInvalidTextureBinding",
                20 => "cudaErrorInvalidChannelDescriptor",
                21 => "cudaErrorInvalidMemcpyDirection",
                22 => "cudaErrorAddressOfConstant",
                23 => "cudaErrorTextureFetchFailed",
                24 => "cudaErrorTextureNotBound",
                25 => "cudaErrorSynchronizationError",
                26 => "cudaErrorInvalidFilterSetting",
                27 => "cudaErrorInvalidNormSetting",
                28 => "cudaErrorMixedDeviceExecution",
                29 => "cudaErrorCudartUnloading",
                30 => "cudaErrorUnknown",
                31 => "cudaErrorNotYetImplemented",
                32 => "cudaErrorMemoryValueTooLarge",
                33 => "cudaErrorInvalidResourceHandle",
                34 => "cudaErrorNotReady",
                35 => "cudaErrorInsufficientDriver",
                36 => "cudaErrorSetOnActiveProcess",
                37 => "cudaErrorInvalidSurface",
                38 => "cudaErrorNoDevice",
                39 => "cudaErrorECCUncorrectable",
                40 => "cudaErrorSharedObjectSymbolNotFound",
                41 => "cudaErrorSharedObjectInitFailed",
                42 => "cudaErrorUnsupportedLimit",
                43 => "cudaErrorDuplicateVariableName",
                44 => "cudaErrorDuplicateTextureName",
                45 => "cudaErrorDuplicateSurfaceName",
                46 => "cudaErrorDevicesUnavailable",
                47 => "cudaErrorInvalidKernelImage",
                48 => "cudaErrorNoKernelImageForDevice",
                49 => "cudaErrorIncompatibleDriverContext",
                50 => "cudaErrorPeerAccessAlreadyEnabled",
                51 => "cudaErrorPeerAccessNotEnabled",
                52 => "cudaErrorDeviceAlreadyInUse",
                53 => "cudaErrorProfilerDisabled",
                54 => "cudaErrorProfilerNotInitialized",
                55 => "cudaErrorProfilerAlreadyStarted",
                56 => "cudaErrorProfilerAlreadyStopped",
                57 => "cudaErrorAssert",
                58 => "cudaErrorTooManyPeers",
                59 => "cudaErrorHostMemoryAlreadyRegistered",
                60 => "cudaErrorHostMemoryNotRegistered",
                61 => "cudaErrorOperatingSystem",
                62 => "cudaErrorPeerAccessUnsupported",
                63 => "cudaErrorLaunchMaxDepthExceeded",
                64 => "cudaErrorLaunchFileScopedTex",
                65 => "cudaErrorLaunchFileScopedSurf",
                66 => "cudaErrorSyncDepthExceeded",
                67 => "cudaErrorLaunchPendingCountExceeded",
                68 => "cudaErrorNotPermitted",
                69 => "cudaErrorNotSupported",
                70 => "cudaErrorHardwareStackError",
                71 => "cudaErrorIllegalInstruction",
                72 => "cudaErrorMisalignedAddress",
                73 => "cudaErrorInvalidAddressSpace",
                74 => "cudaErrorInvalidPc",
                75 => "cudaErrorIllegalAddress",
                76 => "cudaErrorInvalidPtx",
                77 => "cudaErrorInvalidGraphicsContext",
                78 => "cudaErrorNvlinkUncorrectable",
                79 => "cudaErrorJitCompilerNotFound",
                80 => "cudaErrorUnsupportedPtxVersion",
                _ => $"Unknown CUDA error code: {errorCode}"
            };

            Console.WriteLine($"   Error: {errorMessage} ({errorCode})");
        }

        public static void Main(string[] args)
        {
            RunCudaRuntimeDetection();
        }
    }
}