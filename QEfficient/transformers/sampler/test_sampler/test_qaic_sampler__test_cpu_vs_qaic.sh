#!/bin/bash

# Define the configurations
# sequence_lengths=(128)
# batch_sizes=(1 2 8 16)
# vocab_sizes=(1024 4096 32000)
# ctx_lengths=(256 512)
sequence_lengths=(128)
batch_sizes=(1 2 8)
vocab_sizes=(1024)
ctx_lengths=(256 512)

# Ensure the output directory exists
output_dir="./pytest_outputs"
mkdir -p $output_dir

# Initialize counters
success_count=0
total_count=0

# CPU vs QAIC
output_file="$output_dir/test_qaic_sampler__test_cpu_vs_qaic11.log"
rm $output_file

for sequence_length in "${sequence_lengths[@]}"; do
	for batch_size in "${batch_sizes[@]}"; do
		for vocab_size in "${vocab_sizes[@]}"; do
			for ctx_length in "${ctx_lengths[@]}"; do
				pytest_output=$(pytest --disable-warnings -s -v test_qaic_sampler.py::test_cpu_vs_qaic \
					--sequence-length=$sequence_length \
					--batch-size=$batch_size \
					--vocab-size=$vocab_size \
					--ctx-length=$ctx_length \
					--num-devices=1 2>&1 | tee -a $output_file)

				success_count=$((success_count + $(echo "$pytest_output" | grep -c "PASSED")))
				total_count=$((total_count + 1))
			done
		done
	done
done

echo "No. of tests passed: ($success_count/$total_count)" 2>&1 | tee -a $output_file

# # Generate opstats
# rm -r ./outputs_from_qpcs/
# mkdir -p "./outputs_from_qpcs/"
# api_test_cmd="/opt/qti-aic/exec/qaic-api-test -t ./on_device_sampling_qpcs/ -n 10 --aic-profiling-type raw_device_stats --aic-profiling-start-iter 5 --aic-profiling-num-samples 1 --aic-batch-json-input ${output_dir}/aic_batch_io.json --write-output-dir ./outputs_from_qpcs/ -d 11 --aic-profiling-out-dir ./outputs_from_qpcs/"
# echo $api_test_cmd
# $api_test_cmd 2>&1 | tee -a $output_file

# rm -r ./opstats/
# mkdir -p "./opstats/"
# opstats_cmd="/opt/qti-aic/exec/qaic-opstats --qpc ./on_device_sampling_qpcs/programqpc.bin --input-dir ./outputs_from_qpcs/ --output-dir ./opstats/ --summary --trace" 
# echo $opstats_cmd
# $opstats_cmd 2>&1 | tee -a $output_file
