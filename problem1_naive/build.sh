#!/bin/bash
set -e

echo "🔨 Compiling Problem 1: Naive Dense Attention..."
nvcc -std=c++17 -O3 -arch=sm_50 main.cu -o main

if [ $? -eq 0 ]; then
    echo "🎉 Compilation successful! Running attention benchmark..."
    ./main
else
    echo "❌ Compilation failed."
    exit 1
fi
