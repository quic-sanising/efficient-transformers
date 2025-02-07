#!/bin/bash

# Define the configurations
sequence_lengths=(8 16 64 128)
batch_sizes=(1 4 8 16 32)
vocab_sizes=(10 100 1024 2048 4096)
ctx_lengths=(128 512 4096 8192)

# Ensure the output directory exists
output_dir="./pytest_outputs"
mkdir -p $output_dir


# CPU vs vLLM CPU
output_file="$output_dir/test_qaic_sampler__test_cpu_vs_vllm_cpu.txt"
rm $output_file

for sequence_length in "${sequence_lengths[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for vocab_size in "${vocab_sizes[@]}"; do
      for ctx_length in "${ctx_lengths[@]}"; do
        pytest --disable-warnings -s -v test_qaic_sampler.py::test_cpu_vs_vllm_cpu \
          --sequence-length=$sequence_length \
          --batch-size=$batch_size \
          --vocab-size=$vocab_size \
          --ctx-length=$ctx_length 2>&1 | tee -a $output_file
      done
    done
  done
done
