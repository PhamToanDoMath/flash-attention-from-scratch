# 🧩 Problem 4: Fused Attention (End-to-End Kernel)

The goal of this problem is to implement a high-performance **Fused Attention Kernel** that combines shared memory tiling, online streaming softmax, and **warp-level parallel reductions** (`__shfl_sync`) into a single kernel with no intermediate global memory writes.

---

## 📋 High-Performance Concepts

### 1. Fusing Memory Operations
Standard PyTorch attention performs several kernels:
1. Matrix multiplication: $Q K^T$ (writes $N \times N$ to global memory).
2. Softmax (reads $N \times N$, writes $N \times N$ to global memory).
3. Matrix multiplication: $P V$ (reads $N \times N$, writes $N \times D$ to global memory).

Our fused kernel computes the block matrix multiplications and online softmax *locally* within registers and `__shared__` memory. **Only the final output $O$ (shape $N \times D$) is written to global memory.**

### 2. Warp-Level Reductions (`__shfl_sync`)
To find the maximum score ($m$) and the sum of exponents ($d$) across threads within a block, instead of using slow shared memory reductions, we can use high-speed warp shuffle instructions:
- `__shfl_down_sync(mask, val, delta)`: Allows threads in a warp to read registers from threads with a higher lane ID.
- Using a tree-reduction pattern, we can find a warp maximum in just $\log_2(32) = 5$ steps:
  ```cuda
  for (int offset = 16; offset > 0; offset /= 2) {
      val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  ```

---

## 🛠️ Recommended Design Strategy
1. Map a **Thread Block** to handle a group of Query rows (tile size $B_r$).
2. Assign each thread in the block to handle **exactly one row of Query** if $B_r \le 32$ (one warp).
3. Statically allocate shared memory for tiles:
   ```cuda
   __shared__ float s_K[Bc][D];
   __shared__ float s_V[Bc][D];
   ```
4. Stream $K$ and $V$ tiles from global memory into shared memory. At each step, compute similarity scores, perform local streaming softmax updates, and accumulate directly into local registers for the output row.
5. Write the final accumulated output row from registers back to global memory.

---

## 🚀 Kernel Interface

Implement the kernel in `main.cu`:
```cuda
__global__ void fused_attention_kernel(
    const float* Q, // Input Query: Shape (N, D)
    const float* K, // Input Key:   Shape (N, D)
    const float* V, // Input Value: Shape (N, D)
    float* O,       // Output:       Shape (N, D)
    int N, 
    int D
)
```

---

## 🚀 How to Compile & Run

Since your CUDA environment is in WSL, you can compile and run this challenge directly using the provided `build.sh` script.

### Option A: From your WSL Terminal (Recommended)
1. Open WSL and navigate to this folder:
   ```bash
   cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem4_fused
   ```
2. Make the script executable and run it:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

### Option B: From Windows PowerShell
Run the command inside WSL in a single step:
```powershell
wsl bash -c "cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem4_fused && chmod +x build.sh && ./build.sh"
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
Compare the execution speed of this fused kernel against Problems 1, 2, and 3. You should see a massive performance boost as you avoid intermediate memory trips!

