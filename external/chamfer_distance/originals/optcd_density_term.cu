#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void CalculateDensityKernel(
    const float* coordinates,  // [B, N, 3]
    const float* point_cloud,  // [B, M, 3]
    float* density,            // [B, N]
    int B, int N, int M,
    float radius) {
    
    const int batch_idx = blockIdx.x;  // Batch index
    const int coord_idx = threadIdx.x + blockIdx.y * blockDim.x;  // Query coordinate index

    if (batch_idx >= B || coord_idx >= N) return;

    const float* query = &coordinates[(batch_idx * N + coord_idx) * 3];
    const float radius_inv = 1.0 / radius;

    float density_val = 0.0;

    for (int i = 0; i < M; ++i) {
        const float* point = &point_cloud[(batch_idx * M + i) * 3];

        // Compute squared Euclidean distance
        float dx = query[0] - point[0];
        float dy = query[1] - point[1];
        float dz = query[2] - point[2];
        float dist_squared = dx * dx + dy * dy + dz * dz;

        // Compute kernel contribution
        float dist = sqrtf(dist_squared);  // Convert squared distance to actual distance
        density_val += expf(-dist * radius_inv);
    }

    density[batch_idx * N + coord_idx] = density_val;
}

void CalculateDensityLauncher(
    const at::Tensor& coordinates,  // [B, N, 3]
    const at::Tensor& point_cloud,  // [B, M, 3]
    at::Tensor& density,            // [B, N]
    float radius) {

    int B = coordinates.size(0);
    int N = coordinates.size(1);
    int M = point_cloud.size(1);

    const float* coordinates_data = coordinates.data_ptr<float>();
    const float* point_cloud_data = point_cloud.data_ptr<float>();
    float* density_data = density.data_ptr<float>();

    dim3 blocks(B, (N + 255) / 256);
    dim3 threads(256);

    CalculateDensityKernel<<<blocks, threads>>>(
        coordinates_data, point_cloud_data, density_data, B, N, M, radius);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in CalculateDensityLauncher: %s\n", cudaGetErrorString(err));
    }
}
