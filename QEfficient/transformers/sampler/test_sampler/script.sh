# !/bin/bash

# Test penalties
pytest -s -vv test_qaic_sampler_penalties.py::test_cpu_vs_vllm_cpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler_penalties__test_cpu_vs_vllm_cpu.txt
pytest -s -vv test_qaic_sampler_penalties.py::test_cpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler_penalties__test_cpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler_penalties.py::test_gpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler_penalties__test_gpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler_penalties.py::test_gpu_vs_vllm_gpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler_penalties__test_gpu_vs_vllm_gpu.txt

# Test top ks
pytest -s -vv test_qaic_sampler_top_ks.py::test_cpu_vs_vllm_cpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ks__test_cpu_vs_vllm_cpu.txt
pytest -s -vv test_qaic_sampler_top_ks.py::test_cpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ks__test_cpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler_top_ks.py::test_gpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ks__test_gpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler_top_ks.py::test_gpu_vs_vllm_gpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ks__test_gpu_vs_vllm_gpu.txt

# Test top ps
pytest -s -vv test_qaic_sampler_top_ps.py::test_cpu_vs_vllm_cpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ps__test_cpu_vs_vllm_cpu.txt
pytest -s -vv test_qaic_sampler_top_ps.py::test_cpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ps__test_cpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler_top_ps.py::test_gpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ps__test_gpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler_top_ps.py::test_gpu_vs_vllm_gpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler_top_ps__test_gpu_vs_vllm_gpu.txt

# Test end to end
pytest -s -vv test_qaic_sampler.py::test_cpu_vs_vllm_cpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler__test_cpu_vs_vllm_cpu.txt
pytest -s -vv test_qaic_sampler.py::test_cpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler__test_cpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler.py::test_gpu_vs_qaic 2>&1 | tee ./pytest_outputs/test_qaic_sampler__test_gpu_vs_qaic.txt
pytest -s -vv test_qaic_sampler.py::test_gpu_vs_vllm_gpu 2>&1 | tee ./pytest_outputs/test_qaic_sampler__test_gpu_vs_vllm_gpu.txt
