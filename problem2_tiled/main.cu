#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

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

// Compile-time constants (must match the values used in main())
#define HEAD_DIM 64

// Define tile block sizes
#define Bc 24
#define Br 16

// ============================================================================
// 🧩 PROBLEM 2 KERNEL: TILED ATTENTION
// ============================================================================
// Tiled attention with online softmax.
// Each block handles Br rows of Q. We tile over K/V in chunks of Bc columns.
// Uses the "online softmax" trick to avoid needing all scores in shared memory:
//   - Track running max and sum_exp per row across K-tiles
//   - Rescale partial output when a new max is found
// ============================================================================
__global__ void tiled_attention_kernel(
    const float* Q, // Shape: (N, D)
    const float* K, // Shape: (N, D)
    const float* V, // Shape: (N, D)
    float* O,       // Shape: (N, D)
    int N, 
    int D
) {
    // Shared memory tiles — all dimensions are compile-time constants
    __shared__ float s_Q[Br][HEAD_DIM];   // Current Q tile
    __shared__ float s_K[Bc][HEAD_DIM];   // Current K tile
    __shared__ float s_V[Bc][HEAD_DIM];   // Current V tile
    __shared__ float s_S[Br][Bc];         // Tile-local attention scores

    // Each thread handles one row within the tile
    int row = blockIdx.x * Br + threadIdx.x;
    if (row >= N) return;

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    // --- Load Q tile into shared memory ---
    for (int d = 0; d < D; d++) {
        s_Q[threadIdx.x][d] = Q[row * D + d];
    }

    // Per-row accumulators for online softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Per-row output accumulator (in registers)
    float o_acc[HEAD_DIM];
    for (int d = 0; d < D; d++) {
        o_acc[d] = 0.0f;
    }

    // --- Tile over K/V columns ---
    for (int tile = 0; tile < N; tile += Bc) {
        int tile_size = min(Bc, N - tile);  // handle last partial tile

        // Load K tile into shared memory
        if (threadIdx.x < Bc && tile + threadIdx.x < N) {
            for (int d = 0; d < D; d++) {
                s_K[threadIdx.x][d] = K[(tile + threadIdx.x) * D + d];
            }
        }
        // Load V tile into shared memory
        if (threadIdx.x < Bc && tile + threadIdx.x < N) {
            for (int d = 0; d < D; d++) {
                s_V[threadIdx.x][d] = V[(tile + threadIdx.x) * D + d];
            }
        }
        __syncthreads();

        // Compute attention scores for this tile: s_S[threadIdx.x][j] = Q[row] · K[tile+j]
        for (int j = 0; j < tile_size; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += s_Q[threadIdx.x][d] * s_K[j][d];
            } 
            s_S[threadIdx.x][j] = dot * scale;
        }

        // --- Online softmax update ---
        // Find the max in this tile
        float tile_max = -INFINITY;
        for (int j = 0; j < tile_size; j++) {
            tile_max = fmaxf(tile_max, s_S[threadIdx.x][j]);
        }

        // New global max
        float new_max = fmaxf(row_max, tile_max);

        // Rescale previous accumulator: exp(old_max - new_max)
        float rescale = expf(row_max - new_max);
        row_sum *= rescale;
        for (int d = 0; d < D; d++) {
            o_acc[d] *= rescale;
        }

        // Exponentiate tile scores with new max and accumulate
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_size; j++) {
            float p = expf(s_S[threadIdx.x][j] - new_max);
            s_S[threadIdx.x][j] = p;
            tile_sum += p;
        }

        // Accumulate output: O += P_tile @ V_tile
        for (int j = 0; j < tile_size; j++) {
            float p = s_S[threadIdx.x][j];
            for (int d = 0; d < D; d++) {
                o_acc[d] += p * s_V[j][d];
            }
        }

        row_max = new_max;
        row_sum += tile_sum;

        __syncthreads();
    }

    // --- Normalize output by sum of exponentials ---
    float inv_sum = 1.0f / row_sum;
    for (int d = 0; d < D; d++) {
        O[row * D + d] = o_acc[d] * inv_sum;
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
        float row_max = S[i * N + 0];
        for (int j = 1; j < N; ++j) {
            row_max = std::max(row_max, S[i * N + j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = std::exp(S[i * N + j] - row_max);
            sum_exp += S[i * N + j];
        }

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

int main() {
    const int N = 512; // Sequence length
    const int D = 64;  // Head dimension

    std::cout << "🚀 Launching Tiled Attention Arena..." << std::endl;
    std::cout << "Sequence Length (N): " << N << ", Head Dimension (D): " << D << std::endl;
    std::cout << "Tiles Configurations - Br (Row Block): " << Br << ", Bc (Col Block): " << Bc << std::endl;

    float* h_Q = new float[N * D];
    float* h_K = new float[N * D];
    float* h_V = new float[N * D];
    float* h_O_gpu = new float[N * D];
    float* h_O_cpu = new float[N * D];

    srand(42);
    init_matrix(h_Q, N, D);
    init_matrix(h_K, N, D);
    init_matrix(h_V, N, D);

    std::cout << "🖥️ Computing CPU reference attention..." << std::endl;
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, N, D);

    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, N * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, N * D * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, N * D * sizeof(float), cudaMemcpyHostToDevice));

    // Block handles one row-tile of queries
    dim3 threadsPerBlock(Br, 1);
    dim3 numBlocks((N + Br - 1) / Br, 1);

    tiled_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, N, D);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    float diff = verify_results(h_O_gpu, h_O_cpu, N * D);

    std::cout << "📊 Max difference between GPU and CPU: " << diff << std::endl;
    if (diff < 1e-4) {
        std::cout << "✅ [CORRECTNESS PASSED] Your GPU implementation is correct!" << std::endl;
    } else {
        std::cout << "❌ [CORRECTNESS FAILED] Difference exceeds acceptable tolerance of 1e-4!" << std::endl;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        tiled_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, N, D);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "⏱️ Average Execution Time (GPU): " << (milliseconds / iterations) << " ms" << std::endl;

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
