import os
import re
import subprocess

# Define the targets and their files to modify for an N=512, D=64 benchmark
problems = {
    "problem1_naive": {
        "dir": "problem1_naive",
        "file": "problem1_naive/main.cu",
        "replacements": [("const int N = 256;", "const int N = 512;")]
    },
    "problem2_tiled": {
        "dir": "problem2_tiled",
        "file": "problem2_tiled/main.cu",
        "replacements": [("const int N = 256;", "const int N = 512;")]
    },
    "problem3_streaming": {
        "dir": "problem3_streaming",
        "file": "problem3_streaming/main.cu",
        "replacements": [("const int N = 256;", "const int N = 512;")]
    },
    "problem4_fused": {
        "dir": "problem4_fused",
        "file": "problem4_fused/main.cu",
        "replacements": []  # Already N=512, D=64
    },
    "problem5_mha": {
        "dir": "problem5_mha",
        "file": "problem5_mha/main.cu",
        "replacements": []  # Already N=512, D=64, B=8, H=16
    }
}

N = 512
D = 64
B=8
H=16
total_flops = 4 * (N ** 2) * D + 6 * (N ** 2)

results = []

def run_wsl_command(command, cwd):
    try:
        res = subprocess.run(["bash", "-l", "-c", command], capture_output=True, text=True, check=True, cwd=cwd)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command in {cwd}:")
        print(e.stderr)
        return e.stdout

for name, info in problems.items():
    print(f"\n⚡ Preparing benchmark for {name}...")
    filepath = info["file"]
    
    # 1. Read original content
    with open(filepath, "r", encoding="utf-8") as f:
        original_content = f.read()
        
    try:
        # 2. Apply replacements
        content = original_content
        for old, new in info["replacements"]:
            content = content.replace(old, new)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        # 3. Run benchmark via build.sh
        print(f"🚀 Running build.sh for {name}...")
        output = run_wsl_command("./build.sh", os.path.abspath(info["dir"]))
        
        # 4. Parse results
        correctness = "❌ FAILED"
        if "CORRECTNESS PASSED" in output:
            correctness = "✅ PASSED"
            
        time_match = re.search(r"Average Execution Time \(GPU\):\s*([0-9\.]+)\s*ms", output)
        if time_match:
            avg_time = float(time_match.group(1))
            gflops = total_flops / (avg_time * 1e6)
            results.append({
                "problem": name,
                "correctness": correctness,
                "time": f"{avg_time:.4f} ms",
                "gflops": f"{gflops:.3f} GFLOPS"
            })
            print(f"Done! Time: {avg_time:.4f} ms | GFLOPS: {gflops:.3f}")
        else:
            print("Could not find execution time in output.")
            results.append({
                "problem": name,
                "correctness": correctness,
                "time": "N/A",
                "gflops": "N/A"
            })
            
    finally:
        # 5. Restore original file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(original_content)

# Print final report
print("\n" + "="*50)
print("📊 BENCHMARK REPORT (N=512, D=64)")
print("="*50)
print(f"| {'Kernel Name':<20} | {'Correctness':<12} | {'Avg Time (ms)':<15} | {'GFLOPS':<15} |")
print(f"| {'-'*20} | {'-'*12} | {'-'*15} | {'-'*15} |")
for res in results:
    print(f"| {res['problem']:<20} | {res['correctness']:<12} | {res['time']:<15} | {res['gflops']:<15} |")
print("="*50)
