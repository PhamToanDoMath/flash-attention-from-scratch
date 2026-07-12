#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cfloat>
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

#define Bc 32
#define Br 8

// Compile-time constants (must match the values used in main())
#define HEAD_DIM 64

// ============================================================================
// 🧩 PROBLEM 4 KERNEL: FUSED ATTENTION
// ============================================================================
// TODO: Implement the fully fused tiled attention kernel.
// Combine tiling, online softmax, warp reduction, and V multiplication.
// Avoid writing any intermediate NxN values to global memory.
// ============================================================================
__global__ void fused_attention_kernel(
    const float* Q, // Shape: (N, D)
    const float* K, // Shape: (N, D)
    const float* V, // Shape: (N, D)
    float* O,       // Shape: (N, D)
    int N, 
    int D
)
    // ------------------------------------------------------------------------
    // TODO: Write your high-performance fused kernel here.
    // 1. Determine block and thread index.
    // 2. Load current thread's Query row directly into registers or s_Q.
    // 3. Statically allocate s_K and s_V in shared memory.
    // 4. Stream K and V tiles block-by-block.
    // 5. Update running max m, denominator d, and output accumulator O_accum.
    // 6. Write the final outputs back to global memory O.
    // ------------------------------------------------------------------------
{
    // Shared memory tiles — all dimensions are compile-time constants
    __shared__ float s_Q[Br][HEAD_DIM];   // Current Q tile
    __shared__ float s_K[HEAD_DIM][Bc];   // Current K tile
    __shared__ float s_V[Bc][HEAD_DIM];   // Current V tile
    __shared__ float s_S[Br][Bc];         // Tile-local attention scores

    // Each thread handles one row within the tile
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    // --- Load Q tile into shared memory ---
    for (int d = threadIdx.x; d < D; d+=blockDim.x) {
        s_Q[threadIdx.y][d] = Q[row * D + d];
    }
    __syncthreads();
    // Per-row accumulators for online softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Per-row output accumulator (in registers, only 2 elements per thread since Bc=32)
    constexpr int NUM_ELEMENTS_PER_THREAD = HEAD_DIM / 32;
    float o_acc[NUM_ELEMENTS_PER_THREAD] = {0.0f};

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Thread index within the block (0 to 255)
    int num_threads = blockDim.x * blockDim.y;       // 32 * 8 = 256 threads

    // --- Tile over K/V columns ---
    for (int tile = 0; tile < N; tile += Bc) {
        int tile_size = min(Bc, N - tile);  // handle last partial tile

        // Load K,V tile into shared memory
        for (int idx = tid; idx < Bc * D; idx += num_threads) {
            int k_row = idx / D; // row index in key tile (0 to 31)
            int k_col = idx % D; // column index in key tile (0 to 63)
            s_K[k_col][k_row] = K[(tile + k_row) * D + k_col];
            s_V[k_row][k_col] = V[(tile + k_row) * D + k_col];
        }

        // for (int d = threadIdx.x; d < D; d+=blockDim.x) {
            // s_K[threadIdx.y][d] = K[(tile + threadIdx.y) * D + d];
            // s_V[threadIdx.y][d] = V[(tile + threadIdx.y) * D + d];
        // }
        __syncthreads();

        // Compute attention scores for this tile: s_S[threadIdx.y][threadIdx.x] = Q[row] · K[tile + threadIdx.x]
        float dot = 0.0f;
        if (threadIdx.x < tile_size) {
            for (int d = 0; d < D; d++) {
                dot += s_Q[threadIdx.y][d] * s_K[d][threadIdx.x];
            } 
            s_S[threadIdx.y][threadIdx.x] = dot * scale;
        } else{
            s_S[threadIdx.y][threadIdx.x] = -INFINITY; // Zero out padded keys
        }
        __syncthreads();
        // --- Online softmax update ---
        // Find the max in this tile
        float tile_max = s_S[threadIdx.y][threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        // New global max
        float new_max = fmaxf(row_max, tile_max);

        // Rescale previous accumulator: exp(old_max - new_max)
        float rescale = expf(row_max - new_max);
        row_sum *= rescale;
        for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
            o_acc[i] *= rescale;
        }

        // Exponentiate tile scores with new max and accumulate
        float p = 0.0f;
        if (threadIdx.x < tile_size) {
            p = expf(s_S[threadIdx.y][threadIdx.x] - new_max);
            s_S[threadIdx.y][threadIdx.x] = p;
        } else {
            s_S[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Warp-reduce the exponents to get the current tile sum
        float tile_sum = p;
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
        }
        tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

        // Accumulate output: O += P_tile @ V_tile
        for (int j = 0; j < tile_size; j++) {
            float p = s_S[threadIdx.y][j];
            for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
                int d = threadIdx.x + i * 32;
                o_acc[i] += p * s_V[j][d];
            }
        }

        row_max = new_max;
        row_sum += tile_sum;

        __syncthreads();
    }

    // --- Normalize output by sum of exponentials ---
    float inv_sum = 1.0f / row_sum;
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        int d = threadIdx.x + i * 32;
        O[row * D + d] = o_acc[i] * inv_sum;
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

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

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

    std::cout << "🚀 Launching Fused Attention Arena..." << std::endl;
    std::cout << "Sequence Length (N): " << N << ", Head Dimension (D): " << D << std::endl;

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

    dim3 threadsPerBlock(Bc, Br);
    dim3 numBlocks((N + Br - 1) / Br, 1);

    fused_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, N, D);
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
        fused_attention_kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_K, d_V, d_O, N, D);
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
