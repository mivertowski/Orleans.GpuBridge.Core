# Build stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy solution and project files
COPY Orleans.GpuBridge.sln ./
COPY src/Orleans.GpuBridge.Abstractions/*.csproj ./src/Orleans.GpuBridge.Abstractions/
COPY src/Orleans.GpuBridge.Runtime/*.csproj ./src/Orleans.GpuBridge.Runtime/
COPY src/Orleans.GpuBridge.DotCompute/*.csproj ./src/Orleans.GpuBridge.DotCompute/
COPY src/Orleans.GpuBridge.Grains/*.csproj ./src/Orleans.GpuBridge.Grains/
COPY src/Orleans.GpuBridge.BridgeFX/*.csproj ./src/Orleans.GpuBridge.BridgeFX/
COPY tests/Orleans.GpuBridge.Tests/*.csproj ./tests/Orleans.GpuBridge.Tests/

# Restore dependencies
RUN dotnet restore

# Copy source code
COPY . .

# Build and test
RUN dotnet build -c Release --no-restore
RUN dotnet test -c Release --no-build --verbosity normal

# Publish runtime
RUN dotnet publish src/Orleans.GpuBridge.Runtime/Orleans.GpuBridge.Runtime.csproj -c Release -o /app/publish --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/runtime:9.0 AS runtime
WORKDIR /app

# Install GPU drivers and dependencies
RUN apt-get update && apt-get install -y \
    # NVIDIA drivers
    nvidia-driver-525 \
    nvidia-cuda-toolkit \
    # OpenCL
    ocl-icd-opencl-dev \
    # Vulkan
    vulkan-tools \
    libvulkan1 \
    # ROCm for AMD
    rocm-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Copy published app
COPY --from=build /app/publish .

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV AMD_GPU_ENABLED=1
ENV ENABLE_GPU_DIRECT_STORAGE=true

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD dotnet Orleans.GpuBridge.Runtime.dll --health || exit 1

# Run the application
ENTRYPOINT ["dotnet", "Orleans.GpuBridge.Runtime.dll"]