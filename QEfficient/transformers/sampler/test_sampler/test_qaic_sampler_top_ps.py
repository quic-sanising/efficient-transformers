from copy import deepcopy
import subprocess
import torch
import torch.nn as nn

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.sampler.test_sampler.sampler_top_ps import sampler_forward
from QEfficient.transformers.sampler.test_sampler.utils import print_difference_in_tensors
from QEfficient.transformers.sampler.test_sampler.vllm_sampler_topkp import (
    Sampler,
    SamplingMetadata,
)


def test_cpu_vs_vllm_cpu(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"]
    vllm_logits = deepcopy(setup_data_top_ps["logits"]).squeeze(1)

    top_ks = setup_data_top_ps["top_ks"]
    top_ps = setup_data_top_ps["top_ps"]

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=top_ps,
        top_k=top_ks,
        no_top_p=False,
        no_top_k=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
        output_token_ids=None,
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        logits,
        top_ks,
        top_ps,
    )
    vllm_output_logits = vllm_sampler(vllm_logits, sampling_metadata)
    print(f"QEff Output: {qeff_output.logits.squeeze(1)}")
    print(f"VLLM Output: {vllm_output_logits}")

    assert torch.allclose(
        qeff_output.logits.squeeze(1), vllm_output_logits, atol=1e-6
    ), "Output logits do not match"


def test_cpu_vs_qaic(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"]
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data_top_ps["logits"])
    print("QAIC Input Logits", qaic_logits)

    top_ks = setup_data_top_ps["top_ks"]
    qaic_top_ks = deepcopy(setup_data_top_ps["top_ks"])

    top_ps = setup_data_top_ps["top_ps"]
    qaic_top_ps = deepcopy(setup_data_top_ps["top_ps"])

    # ---Run on CPU---
    qeff_start_time = perf_counter()
    qeff_output = sampler_forward(
        None,
        logits.to(torch.float16),
        top_ks,
        top_ps,
    )
    qeff_end_time = perf_counter()
    print("\nOutput\n", qeff_output)
    print(f"Time Taken {(qeff_end_time - qeff_start_time) * 1000: .5f} ms\n")
    
    # ---Run on QAIC---
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    # Export ONNX file
    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_top_ps_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            qaic_logits,
            qaic_top_ks,
            qaic_top_ps,
        ),
        onnx_path,
        input_names=[
            "input_logits",
            "top_ks",
            "top_ps",
        ],
        output_names=[
            "logits",
        ],
        dynamo=False,
        verbose=False,
    )

    # Compile QPC file
    qpc_dir_path = "./on_device_sampling_qpcs/"
    compile_cmd = [
        "/opt/qti-aic/exec/qaic-exec",
        "-v",
        "-aic-hw",
        "-convert-to-fp16",
        "-aic-num-cores=14",
        "-aic-num-of-instances=1",
        "-num-iter=100",
        f"-m={onnx_path}",
        "-stats-level=70",
        "-aic-pmu-recipe=KernelUtil",
        "-ddr-stats",
        "-time-passes",
        "-mxfp6-matmul",
        "-aic-enable-depth-first",
        "-retained-state=true",
        "-aic-perf-metrics",
        f"-aic-binary-dir={qpc_dir_path}",
        "-aic-hw-version=2.0",
        "-compile-only",
    ]
    subprocess.run(["rm", "-rf", f"{qpc_dir_path}"])
    print("Compile command", " ".join(compile_cmd))
    result = subprocess.run(compile_cmd, capture_output=True, text=True)    
    # print(result.stdout)
    if (result.returncode != 0):
        print(result.stderr)

    # Run QPC file
    session = QAICInferenceSession(qpc_path=qpc_dir_path, device_ids=[0], enable_debug_logs=False)
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "top_ks": qaic_top_ks.detach().cpu().numpy(),
        "top_ps": qaic_top_ps.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    qaic_start_time = perf_counter()
    outputs = session.run(inputs)
    qaic_end_time = perf_counter()
    print("\nQAIC Output\n", outputs)
    print(f"Time Taken {(qaic_end_time - qaic_start_time) * 1000: .5f} ms\n")

    hw_output_logits = torch.from_numpy(outputs["logits"])

    print("\nLogits\n", qeff_output.logits)
    print("\nQAIC Logits\n", hw_output_logits)

    # Compare outputs
    assert torch.allclose(
        qeff_output.logits.to(torch.float32), hw_output_logits, atol=1e-4
    ), print_difference_in_tensors(
        qeff_output.logits.to(torch.float32),
        "Logits",
        hw_output_logits,
        "QAIC Logits",
        1e-4,
    )


def test_gpu_vs_qaic(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"].cuda()
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data_top_ps["logits"])
    print("QAIC Input Logits", qaic_logits)

    top_ks = setup_data_top_ps["top_ks"].cuda()
    qaic_top_ks = deepcopy(setup_data_top_ps["top_ks"])

    top_ps = setup_data_top_ps["top_ps"].cuda()
    qaic_top_ps = deepcopy(setup_data_top_ps["top_ps"])

    # ---Run on CPU---
    qeff_start_time = perf_counter()
    qeff_output = sampler_forward(
        None,
        logits.to(torch.float16),
        top_ks,
        top_ps,
    )
    qeff_end_time = perf_counter()
    print("\nOutput\n", qeff_output)
    print(f"Time Taken {(qeff_end_time - qeff_start_time) * 1000: .5f} ms\n")
    
    # ---Run on QAIC---
    class Sampler(nn.Module):
        def __init__(self):
            super(Sampler, self).__init__()
            self.forward = sampler_forward

    # Export ONNX file
    model = Sampler()
    onnx_path = "./on_device_sampling_onnx/test_sampler_top_ps_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            qaic_logits,
            qaic_top_ks,
            qaic_top_ps,
        ),
        onnx_path,
        input_names=[
            "input_logits",
            "top_ks",
            "top_ps",
        ],
        output_names=[
            "logits",
        ],
        dynamo=False,
        verbose=False,
    )

    # Compile QPC file
    qpc_dir_path = "./on_device_sampling_qpcs/"
    compile_cmd = [
        "/opt/qti-aic/exec/qaic-exec",
        "-v",
        "-aic-hw",
        "-convert-to-fp16",
        "-aic-num-cores=14",
        "-aic-num-of-instances=1",
        "-num-iter=100",
        f"-m={onnx_path}",
        "-stats-level=70",
        "-aic-pmu-recipe=KernelUtil",
        "-ddr-stats",
        "-time-passes",
        "-mxfp6-matmul",
        "-aic-enable-depth-first",
        "-retained-state=true",
        "-aic-perf-metrics",
        f"-aic-binary-dir={qpc_dir_path}",
        "-aic-hw-version=2.0",
        "-compile-only",
    ]
    subprocess.run(["rm", "-rf", f"{qpc_dir_path}"])
    print("Compile command", " ".join(compile_cmd))
    result = subprocess.run(compile_cmd, capture_output=True, text=True)    
    # print(result.stdout)
    if (result.returncode != 0):
        print(result.stderr)

    # Run QPC file
    session = QAICInferenceSession(qpc_path=qpc_dir_path, device_ids=[0], enable_debug_logs=False)
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "top_ks": qaic_top_ks.detach().cpu().numpy(),
        "top_ps": qaic_top_ps.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    qaic_start_time = perf_counter()
    outputs = session.run(inputs)
    qaic_end_time = perf_counter()
    print("\nQAIC Output\n", outputs)
    print(f"Time Taken {(qaic_end_time - qaic_start_time) * 1000: .5f} ms\n")
    
    hw_output_logits = torch.from_numpy(outputs["logits"])

    print("\nLogits\n", qeff_output.logits)
    print("\nQAIC Logits\n", hw_output_logits)

    # Compare outputs
    assert torch.allclose(
        qeff_output.logits.cpu().to(torch.float32), hw_output_logits, atol=1e-4
    ), print_difference_in_tensors(
        qeff_output.logits.cpu().to(torch.float32),
        "Logits",
        hw_output_logits,
        "QAIC Logits",
        1e-4,
    )


def test_gpu_vs_vllm_gpu(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"].cuda()
    vllm_logits = deepcopy(setup_data_top_ps["logits"]).squeeze(1).cuda()

    top_ks = setup_data_top_ps["top_ks"].cuda()
    top_ps = setup_data_top_ps["top_ps"].cuda()

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=top_ps,
        top_k=top_ks,
        no_top_p=False,
        no_top_k=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=None,
        presence_penalties=None,
        repetition_penalties=None,
        output_token_ids=None,
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        logits,
        top_ks,
        top_ps,
    )
    vllm_output_logits = vllm_sampler(vllm_logits, sampling_metadata)
    print(f"QEff Output: {qeff_output.logits.squeeze(1)}")
    print(f"VLLM Output: {vllm_output_logits}")

    assert torch.allclose(
        qeff_output.logits.squeeze(1).cpu(), vllm_output_logits.cpu(), atol=1e-6
    ), "Output logits do not match"
