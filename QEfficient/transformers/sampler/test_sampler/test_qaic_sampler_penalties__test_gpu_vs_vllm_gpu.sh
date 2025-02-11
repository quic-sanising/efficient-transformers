#!/bin/bash

# Define the configurations
sequence_lengths=(8 16 64 128)
batch_sizes=(1 4 8 16 32)
vocab_sizes=(10 100 1024 2048 4096)
ctx_lengths=(128 512 4096 8192)

# Ensure the output directory exists
output_dir="./pytest_outputs"
mkdir -p $output_dir

# Initialize counters
success_count=0
total_count=0

# GPU vs vLLM GPU
output_file="$output_dir/test_qaic_sampler_penalties__test_gpu_vs_vllm_gpu.txt"
rm $output_file

for sequence_length in "${sequence_lengths[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for vocab_size in "${vocab_sizes[@]}"; do
            for ctx_length in "${ctx_lengths[@]}"; do
                pytest_output=$(pytest --disable-warnings -s -v test_qaic_sampler_penalties.py::test_gpu_vs_vllm_gpu \
                    --sequence-length=$sequence_length \
                    --batch-size=$batch_size \
                    --vocab-size=$vocab_size \
                    --ctx-length=$ctx_length 2>&1 | tee -a $output_file)

                success_count=$((success_count + $(echo "$pytest_output" | grep -c "PASSED")))
                total_count=$((total_count + 1))
            done
        done
    done
done

echo "No. of tests passed: ($success_count/$total_count)" 2>&1 | tee -a $output_file
