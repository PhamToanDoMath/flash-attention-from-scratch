# 🧩 Problem 2: Tiled Attention (Memory Optimization)

The goal of this problem is to use **Shared Memory Tiling** to optimize memory throughput. 

In attention, the similarity matrix $S = QK^T$ requires loading elements repeatedly from global memory. By dividing the sequence dimension $N$ into tiles of size `BLOCK_SIZE` and loading these tiles into high-speed `__shared__` memory, we can reuse data across threads and drastically reduce global memory traffic.

---

## 📋 Mathematical Requirement & Tiling Concept

We divide the sequence length $N$ into blocks of size `B_r` (block size for Query row dimension) and `B_c` (block size for Key/Value column dimension).

For each block:
1. **Load Query Tile**: Load a tile of $Q$ (shape $B_r \times D$) into shared memory `s_Q`.
2. **Loop Over Key Tiles**: For each step $j$ from $0$ to $N/B_c$:
   - Load a tile of $K$ (shape $B_c \times D$) into shared memory `s_K`.
   - Compute partial dot products between `s_Q` and `s_K`, and accumulate them in a block-wide matrix $S_{local}$.
3. **Compute Softmax**: Apply softmax row-wise across the full $N$ dimension for the active block.
4. **Multiply by Value Tile**:
   - Load a tile of $V$ (shape $B_c \times D$) into shared memory `s_V`.
   - Multiply the active softmax probabilities by `s_V` and accumulate into the output tile of $O$ (shape $B_r \times D$).
5. **Write Back**: Write the final output tile $O$ back to global memory.

---

## 🛠️ Thread Mapping & Shared Memory

1. **Grid and Blocks**: 
   - Set the block size to `(B_r, 1)` or standard 2D layouts. Since $D$ is small (e.g. 64), a simple layout is `blockDim = (B_r, 1)`.
   - Use `extern __shared__ float s_mem[]` or statically declared arrays inside the kernel:
     ```cuda
     __shared__ float s_Q[B_r][D];
     __shared__ float s_K[B_c][D];
     __shared__ float s_V[B_c][D];
     ```
2. **Synchronization**: Always call `__syncthreads()` after loading data into shared memory before reading it, and after reading it before loading the next tile!

---

## 🚀 Kernel Interface

Implement the kernel in `main.cu`:
```cuda
__global__ void tiled_attention_kernel(
    const float* Q, // Input Query: Shape (N, D), row-major
    const float* K, // Input Key:   Shape (N, D), row-major
    const float* V, // Input Value: Shape (N, D), row-major
    float* O,       // Output:       Shape (N, D), row-major
    int N,          // Sequence length
    int D           // Head dimension
)
```

---

## 🚀 How to Compile & Run

Since your CUDA environment is in WSL, you can compile and run this challenge directly using the provided `build.sh` script.

### Option A: From your WSL Terminal (Recommended)
1. Open WSL and navigate to this folder:
   ```bash
   cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem2_tiled
   ```
2. Make the script executable and run it:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

### Option B: From Windows PowerShell
Run the command inside WSL in a single step:
```powershell
wsl bash -c "cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem2_tiled && chmod +x build.sh && ./build.sh"
```

---

## 💡 GPU Architecture Customization
The `build.sh` script is configured with:
```bash
nvcc -O3 -arch=sm_80 main.cu -o main
```
* If you are running an **Ampere GPU** (e.g., RTX 30xx, A100), keep `-arch=sm_80`.
* For **Turing GPUs** (e.g., RTX 20xx, GTX 1660), change to `-arch=sm_75`.
* For **Ada Lovelace GPUs** (e.g., RTX 40xx), change to `-arch=sm_89`.
* For **Hopper GPUs** (e.g., H100), change to `-arch=sm_90`.

---

## 🎯 Verification and Benchmarking
The harness will run a CPU-side reference attention implementation, copy your GPU results, verify correctness down to $10^{-4}$ precision, and then output:
- **Max difference between GPU and CPU** (must be $< 10^{-4}$ to pass!).
- **Average Execution Time (GPU)** in milliseconds. Compare this with Problem 1 to observe the effect of memory tiling!

