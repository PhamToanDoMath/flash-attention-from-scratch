# 🧩 Problem 3: Streaming Softmax (FlashAttention Core)

This is the core mathematical breakthrough of **FlashAttention**. 

Normally, to compute softmax over a row $S$, you need to see the entire row to find the maximum value $m = \max_j S_j$, then compute $\sum_j \exp(S_j - m)$. This forces you to write the $N \times N$ matrix to global memory. 

**Online (Streaming) Softmax** allows you to compute the exact softmax values dynamically, tile-by-tile, while streaming through Key and Value blocks, dropping the memory footprint from $O(N^2)$ to $O(N)$!

---

## 📋 The Mathematics of Online Softmax

Suppose we compute attention row-wise for a single query row.
Let $S$ be the similarity vector of length $N$. We slice this vector into blocks.

For a current block index $k$, let the local scores in that block be $S^{(k)}$.

We maintain three running variables for the query row:
1. $m^{(k)}$: The running maximum of scores seen so far.
2. $d^{(k)}$: The running denominator (normalizer sum) of exponents seen so far.
3. $O^{(k)}$: The running output vector (dimension $D$) accumulated so far.

### Initialization (Step 0)
- $m^{(0)} = -\infty$
- $d^{(0)} = 0$
- $O^{(0)} = 0$ (vector of length $D$)

### Update Step (From block $k-1$ to block $k$)
Given the new tile of scores $S^{(k)}$ of size $B_c$:
1. **Find Local Max**: 
   $$m_{\text{local}} = \max_{j} S^{(k)}_j$$
2. **Compute New Global Max**: 
   $$m^{(k)} = \max(m^{(k-1)}, m_{\text{local}})$$
3. **Rescale Denominators**:
   Compute the local sum of exponents shifted by the *new* running maximum $m^{(k)}$:
   $$d_{\text{local}} = \sum_{j} \exp(S^{(k)}_j - m^{(k)})$$
   Rescale the previous normalizer and add the new one:
   $$d^{(k)} = d^{(k-1)} \cdot \exp(m^{(k-1)} - m^{(k)}) + d_{\text{local}}$$
4. **Rescale and Accumulate Output Vector**:
   The output accumulated so far ($O^{(k-1)}$) was divided by $d^{(k-1)}$ and shifted by $m^{(k-1)}$. We must rescale it to align with the new denominator $d^{(k)}$ and new maximum $m^{(k)}$:
   $$O^{(k)} = O^{(k-1)} \cdot \left( \frac{d^{(k-1)} \cdot \exp(m^{(k-1)} - m^{(k)})}{d^{(k)}} \right) + \sum_{j} \left( \frac{\exp(S^{(k)}_j - m^{(k)})}{d^{(k)}} \right) V_j$$

Once we have streamed through all blocks of Key and Value, the final vector $O^{(last)}$ is **mathematically identical** to standard softmax attention output, but computed without ever writing a single $N \times N$ matrix element to global memory!

---

## 🚀 Kernel Interface

Implement the kernel in `main.cu`:
```cuda
__global__ void streaming_softmax_attention_kernel(
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
   cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem3_streaming
   ```
2. Make the script executable and run it:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

### Option B: From Windows PowerShell
Run the command inside WSL in a single step:
```powershell
wsl bash -c "cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem3_streaming && chmod +x build.sh && ./build.sh"
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
If your running maxes and scaling equations are mathematically correct, they will match the dense CPU attention exactly!

