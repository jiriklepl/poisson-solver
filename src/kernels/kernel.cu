#include "poisson-solver/kernel.cuh"

namespace poisson_solver {
__global__ void VectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

void RunVectorAdd() {
    constexpr int size = 50000;
    std::vector<float> a(size, 3.1f);
    std::vector<float> b(size, 2.0f);
    std::vector<float> c(size, 0.0f);

    float *cuA;
    float *cuB;
    float *cuC;

    cudaMalloc(&cuA, size * sizeof(float));
    cudaMalloc(&cuB, size * sizeof(float));
    cudaMalloc(&cuC, size * sizeof(float));

    cudaMemcpy(cuA, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuB, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of "
              << threadsPerBlock << " threads\n";

    VectorAdd<<<blocksPerGrid, threadsPerBlock>>>(cuA, cuB, cuC, size);

    cudaMemcpy(c.data(), cuC, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < size; ++i) {
        if (std::abs(c[i] - (a[i] + b[i])) > 1e-5) {
            std::cout << "Error: C = " << c[i] << "\n";
            return;
        }
    }
}
}  // namespace poisson_solver
