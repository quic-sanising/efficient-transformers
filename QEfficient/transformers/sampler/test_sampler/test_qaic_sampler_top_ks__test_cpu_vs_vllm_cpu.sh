#!/bin/bash

# Define the configurations
batch_sizes=(1 4 8 16 32)
vocab_sizes=(10 100 1024 2048 4096)

# Ensure the output directory exists
output_dir="./pytest_outputs"
mkdir -p $output_dir

# Initialize counters
success_count=0
total_count=0

# CPU vs vLLM CPU
output_file="$output_dir/test_qaic_sampler_top_ks__test_cpu_vs_vllm_cpu.txt"
rm $output_file

for batch_size in "${batch_sizes[@]}"; do
    for vocab_size in "${vocab_sizes[@]}"; do
        pytest_output=$(pytest --disable-warnings -s -v test_qaic_sampler_top_ks.py::test_cpu_vs_vllm_cpu \
            --batch-size=$batch_size \
            --vocab-size=$vocab_size 2>&1 | tee -a $output_file)

        success_count=$((success_count + $(echo "$pytest_output" | grep -c "PASSED")))
        total_count=$((total_count + 1))
    done
done

echo "No. of tests passed: ($success_count/$total_count)" 2>&1 | tee -a $output_file
