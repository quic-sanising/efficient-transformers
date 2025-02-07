#!/bin/bash

# Define the configurations
batch_sizes=(1 4 8 16 32)
vocab_sizes=(10 100 1024 2048 4096)

# Ensure the output directory exists
output_dir="./pytest_outputs"
mkdir -p $output_dir

# GPU vs vLLM GPU
output_file="$output_dir/test_qaic_sampler_top_ps__test_gpu_vs_vllm_gpu.txt"
rm $output_file

for batch_size in "${batch_sizes[@]}"; do
    for vocab_size in "${vocab_sizes[@]}"; do
        pytest --disable-warnings -s -v test_qaic_sampler_top_ps.py::test_gpu_vs_vllm_gpu \
            --batch-size=$batch_size \
            --vocab-size=$vocab_size 2>&1 | tee -a $output_file
    done
done
