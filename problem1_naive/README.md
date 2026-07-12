# 🧩 Problem 1: Naive Dense Attention

The goal of this first problem is to implement standard Softmax-based attention directly, establishing a solid baseline for correctness and numerical checks.

## 📋 Mathematical Requirement

Given matrices $Q, K, V \in \mathbb{R}^{N \times D}$ (where $N$ is sequence length and $D$ is head dimension):

1. **Scale Dot-Product Similarity**: Compute the $N \times N$ similarity matrix $S$:
   $$S_{i, j} = \frac{1}{\sqrt{D}} \sum_{k=0}^{D-1} Q_{i, k} K_{j, k}$$

2. **Row-wise Softmax**: Apply softmax row-wise to get the attention weights $P \in \mathbb{R}^{N \times N}$:
   $$P_{i, j} = \frac{\exp(S_{i, j} - m_i)}{\sum_{l=0}^{N-1} \exp(S_{i, l} - m_i)}$$
   *Note: Subtract the row maximum $m_i = \max_l(S_{i, l})$ before exponentiation for numerical stability.*

3. **Value Accumulation**: Multiply the attention weights by $V$ to get the output $O \in \mathbb{R}^{N \times D}$:
   $$O_{i, k} = \sum_{j=0}^{N-1} P_{i, j} V_{j, k}$$

---

## 🛠️ Thread Mapping & Constraints
In this naive implementation:
- You may use a simple block/thread configuration. A straightforward choice is mapping **one thread per query row** ($i \in [0, N-1]$), or **one thread block per query row** where threads in a block collaborate to compute the row-wise softmax and output accumulation.
- We recommend mapping **one thread to compute one output element $O_{i, k}$** or **one thread per query row** for ease of development.
- **Constraints**: No optimizations (no shared memory, no tiling). Let every thread read directly from global memory as needed.

---

## 🚀 Kernel Interface

Implement the kernel in `main.cu`:
```cuda
__global__ void naive_attention_kernel(
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
   cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem1_naive
   ```
2. Make the script executable and run it:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

### Option B: From Windows PowerShell
Run the command inside WSL in a single step:
```powershell
wsl bash -c "cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem1_naive && chmod +x build.sh && ./build.sh"
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
- **Average Execution Time (GPU)** in milliseconds.

