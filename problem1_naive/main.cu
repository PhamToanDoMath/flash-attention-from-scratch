#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/std/cmath>


// Helper to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// 🧩 PROBLEM 1 KERNEL: NAIVE DENSE ATTENTION
// ============================================================================
// TODO: Implement your naive attention kernel here.
// Each query row i does:
// 1. Compute similarity scores S[i, j] = Q[i, :] * K[j, :] / sqrt(D) for all j.
// 2. Perform row-wise softmax over S[i, :] (subtract max score for stability!).
// 3. Compute output O[i, k] = Sum_j (P[i, j] * V[j, k]).
// ============================================================================
__global__ void naive_attention_kernel(
    const float* Q, // Shape: (N, D)
    const float* K, // Shape: (N, D)
    const float* V, // Shape: (N, D)
    float* O,       // Shape: (N, D)
    int N, 
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N){
        return;
    }
    // Use a fixed-size local stack array (supports sequence length N up to 1024).
    // Stack memory is hundreds of times faster than GPU heap allocation (new/delete)
    // and is 100% robust across all CUDA devices and WSL configurations.
    float S[1024];
    if (N > 1024) return; // Guard against out-of-bounds
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    // Compute similarity scores of each row
    for (int j = 0; j < N; j++){
        float sum = 0;
        for (int k = 0; k < D; k++){
            sum += Q[i*D + k] * K[j*D + k];
        }
        S[j] = sum * scale;
    }
    
    // Compute row maximum and row wise softmax
    float row_max = S[0];
    for (int j = 1; j < N; ++j) {
        row_max = fmaxf(row_max, S[j]);
    }
    float sum_exp = 0;
    for (int j = 0; j < N; j++){
        sum_exp += expf(S[j] - row_max);
    }
    for (int j = 0; j < N; j++){
        S[j] = expf(S[j] - row_max) / sum_exp;
    }
    // Compute value accumulation
    for (int k = 0; k < D; k++){
        float sum = 0;
        for (int j = 0; j < N; j++){
            sum += S[j] * V[j*D + k];
        }
        O[i*D + k] = sum;
    }
}

// ============================================================================
// 🖥️ CPU Reference Implementation (Do Not Modify)
// ============================================================================
void cpu_attention(const float* Q, const float* K, const float* V, float* O, int N, int D) {
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    float* S = new float[N * N];

    // 1. Compute Q @ K^T / sqrt(D)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += Q[i * D + k] * K[j * D + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // 2. Row-wise Softmax
    for (int i = 0; i < N; ++i) {
        // Find maximum in the row for numerical stability
        float row_max = S[i * N + 0];
        for (int j = 1; j < N; ++j) {
            row_max = std::max(row_max, S[i * N + j]);
        }

        // Compute exponent and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = std::exp(S[i * N + j] - row_max);
            sum_exp += S[i * N + j];
        }

        // Divide by sum of exponents (normalize)
        for (int j = 0; j < N; ++j) {
            S[i * N + j] /= sum_exp;
        }
    }

    // 3. Multiply by V: O = S @ V
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < D; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += S[i * N + j] * V[j * D + k];
            }
            O[i * D + k] = sum;
        }
    }

    delete[] S;
}

// Initialize matrices with random values
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Compare two matrices and find max difference
float verify_results(const float* gpu_res, const float* cpu_res, int total_elements) {
    float max_diff = 0.0f;
    for (int i = 0; i < total_elements; ++i) {
        float diff = std::abs(gpu_res[i] - cpu_res[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

// Print matrix with formatted columns
void print_matrix(const char* name, const float* mat, int rows, int cols) {
    std::cout << "--- Matrix: " << name << " (" << rows << "x" << cols << ") ---" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::printf("%8.4f ", mat[i * cols + j]);
        }
        std::cout << std::endl;
    }
    std::cout << "---------------------------------------" << std::endl;
}

int main() {
    // Problem parameters
    const int N = 512; // Sequence length
    const int D = 64;  // Head dimension

    std::cout << "🚀 Launching Naive Attention Arena..." << std::endl;
    std::cout << "Sequence Length (N): " << N << ", Head Dimension (D): " << D << std::endl;

    // Allocate host memory
    float* h_Q = new float[N * D];
    float* h_K = new float[N * D];
    float* h_V = new float[N * D];
    float* h_O_gpu = new float[N * D];
    float* h_O_cpu = new float[N * D];

    // Initialize inputs
    srand(42);
    init_matrix(h_Q, N, D);
    init_matrix(h_K, N, D);
    init_matrix(h_V, N, D);

    // Compute CPU reference
    std::cout << "🖥️ Computing CPU reference attention..." << std::endl;
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, N, D);

    // Allocate GPU memory
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, N * D * sizeof(float)));

    // Copy to Device
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, N * D * sizeof(float), cudaMemcpyHostToDevice));

    // Configure Grid and Block dimensions
    // TODO: Change grid/block configuration if your kernel needs it!
    dim3 threadsPerBlock(32, 1);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    // Warm-up run
    CHECK_CUDA(cudaMemset(d_O, 0xFF, N * D * sizeof(float))); // Pre-fill device buffer with NaN bytes
    
    naive_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, N, D);
    CHECK_CUDA(cudaGetLastError()); // Catch silent launch errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // Pre-fill host buffer with 777.0 to confirm cudaMemcpy actually runs
    for (int idx = 0; idx < N * D; ++idx) h_O_gpu[idx] = 777.0f;

    // Copy back and initial check
    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    float diff = verify_results(h_O_gpu, h_O_cpu, N * D);

    std::cout << "📊 Max difference between GPU and CPU: " << diff << std::endl;
    if (diff < 1e-4) {
        std::cout << "✅ [CORRECTNESS PASSED] Your GPU implementation is correct!" << std::endl;
    } else {
        std::cout << "❌ [CORRECTNESS FAILED] Difference exceeds acceptable tolerance of 1e-4!" << std::endl;
        print_matrix("GPU Output (h_O_gpu)", h_O_gpu, N, D);
        print_matrix("CPU Reference (h_O_cpu)", h_O_cpu, N, D);
    }

    // Benchmark Execution Time using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        naive_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, N, D);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "⏱️ Average Execution Time (GPU): " << (milliseconds / iterations) << " ms" << std::endl;

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_gpu;
    delete[] h_O_cpu;

    return 0;
}
