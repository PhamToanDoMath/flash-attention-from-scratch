#!/bin/bash
set -e

echo "🔨 Compiling Problem 2: Tiled Attention..."
nvcc -std=c++17 -O3 -arch=sm_50 main.cu -o main

if [ $? -eq 0 ]; then
    echo "🎉 Compilation successful! Running tiled attention benchmark..."
    ./main
else
    echo "❌ Compilation failed."
    exit 1
fi
