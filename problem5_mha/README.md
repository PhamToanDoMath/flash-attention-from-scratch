# 🧩 Problem 5: Multi-Head Attention Extension

The goal of this final problem is to generalize your high-performance fused attention kernel to support batched **Multi-Head Attention (MHA)**. 

In actual deep learning models (like Transformers), attention is never done on a single sequence row. Instead, inputs are batches of sequences processed through multiple heads in parallel.

---

## 📋 Batched Multi-Head Attention Structure

The shape of the inputs and outputs increases to 4D tensors:
- **$Q$ (Query)**: Shape $(B, H, N, D)$
- **$K$ (Key)**: Shape $(B, H, N, D)$
- **$V$ (Value)**: Shape $(B, H, N, D)$
- **$O$ (Output)**: Shape $(B, H, N, D)$

Where:
- $B$: Batch size (e.g. number of sentences).
- $H$: Number of attention heads.
- $N$: Sequence length.
- $D$: Head dimension.

Each head of each batch behaves as an independent attention block:
$$O_{b, h, :, :} = \text{softmax}\left(\frac{Q_{b, h, :, :} K_{b, h, :, :}^T}{\sqrt{D}}\right) V_{b, h, :, :}$$

---

## 🛠️ Thread and Grid Offsets

To scale your kernel, map:
- `blockIdx.y` to the batch index ($b \in [0, B-1]$).
- `blockIdx.z` to the head index ($h \in [0, H-1]$).
- `blockIdx.x` to the query row tile (same as single head).

### Computing Memory Offsets
Inside the kernel, you will need to offset your $Q$, $K$, $V$, and $O$ pointers:
```cuda
int b = blockIdx.y;
int h = blockIdx.z;

// Offset to the start of the current batch and head
int batch_head_offset = b * (H * N * D) + h * (N * D);

const float* Q_offset = Q + batch_head_offset;
const float* K_offset = K + batch_head_offset;
const float* V_offset = V + batch_head_offset;
float* O_offset       = O + batch_head_offset;
```
Once you offset the base pointers, the rest of your kernel logic is **exactly identical** to the fused kernel from Problem 4!

---

## 🚀 Kernel Interface

Implement the kernel in `main.cu`:
```cuda
__global__ void mha_attention_kernel(
    const float* Q, // Input Query: Shape (B, H, N, D)
    const float* K, // Input Key:   Shape (B, H, N, D)
    const float* V, // Input Value: Shape (B, H, N, D)
    float* O,       // Output:       Shape (B, H, N, D)
    int B,          // Batch size
    int H,          // Number of heads
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
   cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem5_mha
   ```
2. Make the script executable and run it:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

### Option B: From Windows PowerShell
Run the command inside WSL in a single step:
```powershell
wsl bash -c "cd /mnt/c/Users/phamt/Desktop/projects/CUDA_things/problem5_mha && chmod +x build.sh && ./build.sh"
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
The harness will run a batched 4D CPU Multi-Head Attention reference, copy your GPU results, verify correctness down to $10^{-4}$ precision, and then output:
- **Max difference between GPU and CPU** (must be $< 10^{-4}$ to pass!).
- **Average Execution Time (GPU MHA)** in milliseconds.
Congratulations on building a complete, high-performance, batched Multi-Head FlashAttention kernel from scratch!

