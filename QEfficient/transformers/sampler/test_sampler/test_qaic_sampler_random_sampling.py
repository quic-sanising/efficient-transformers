from copy import deepcopy
from time import perf_counter
import numpy as np
import subprocess
import torch
import torch.nn as nn

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.sampler.test_sampler.sampler_random_sampling import sampler_forward
from QEfficient.transformers.sampler.test_sampler.utils import (
    print_difference_in_tensors,
    get_summary_statistics,
    get_kl_divergence,
    get_z_score,
)
from QEfficient.transformers.sampler.test_sampler.vllm_sampler_random_sampling import (
    Sampler,
    SamplingMetadata,
)


def test_cpu_vs_vllm_cpu(setup_data_random_sampling):
    print(setup_data_random_sampling["seed"])

    logits = setup_data_random_sampling["logits"]
    vllm_logits = deepcopy(setup_data_random_sampling["logits"]).squeeze(1)

    # temperatures = setup_data_random_sampling["temperatures"]
    random_numbers = setup_data_random_sampling["random_numbers"]
    generators = setup_data_random_sampling["generators"]

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        no_top_p=True,
        no_top_k=True,
        no_min_p=True,
        generators=generators,
        max_num_logprobs=0,
        no_penalties=True,
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
        # temperatures,
        random_numbers,
    )
    vllm_output = vllm_sampler(vllm_logits, sampling_metadata)

    print("\nLogits\n", qeff_output[0])
    print("\nvLLM Logits\n", vllm_output[0])

    print("\nProbs\n", qeff_output[1])
    print("\nvLLM Probs\n", vllm_output[1])

    print("\nNext Tokens\n", qeff_output[2])
    print("\nvLLM Next Tokens\n", vllm_output[2])

    # Compare outputs
    has_failed = False
    if not torch.allclose(qeff_output[0].squeeze(1), vllm_output[0], atol=1e-4):
        print_difference_in_tensors(
            qeff_output[0].squeeze(1),
            "Logits",
            vllm_output[0],
            "vLLM Logits",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(qeff_output[1].squeeze(1), vllm_output[1], atol=1e-4):
        print_difference_in_tensors(
            qeff_output[1].squeeze(1),
            "Probs",
            vllm_output[1],
            "vLLM Probs",
            1e-4,
        )
        has_failed = True

    # Summary Stats (Mean, Variance, ...)
    qeff_sumarry_stats = get_summary_statistics(qeff_output[2].squeeze(2).squeeze(1))
    vllm_summary_stats = get_summary_statistics(vllm_output[2])

    for k in qeff_sumarry_stats.keys():
        print("\n")
        print(f"Next Tokens {k}: {qeff_sumarry_stats[k]}")
        print(f"vLLM Next Tokens {k}: {vllm_summary_stats[k]}")

        if not torch.allclose(qeff_sumarry_stats[k], vllm_summary_stats[k], atol=1):
            print_difference_in_tensors(
                qeff_sumarry_stats[k],
                f"Next Tokens {k}",
                vllm_summary_stats[k],
                f"vLLM Next Tokens {k}",
                0.1,
            )
            has_failed = True

    # KL Divergence
    kl_divergence = get_kl_divergence(
        qeff_output[2].squeeze(2).squeeze(1), vllm_output[2], logits.shape[-1]
    )
    print("\n\nKL Divergence", kl_divergence)
    if kl_divergence > 0.1:
        has_failed = True

    # Z Test
    z_score, p_value = get_z_score(
        qeff_output[2].squeeze(2).squeeze(1),
        vllm_output[2],
        qeff_output[2].shape[0],
        vllm_output[2].shape[0],
    )
    print("\n\nZ Score", z_score)
    print("P Value", p_value)
    if p_value < 0.05:
        has_failed = True

    assert has_failed == False, "Test Failed"


def test_cpu_vs_qaic(setup_data_random_sampling):
    print(setup_data_random_sampling["seed"])

    logits = setup_data_random_sampling["logits"]
    qaic_logits = deepcopy(setup_data_random_sampling["logits"])

    # temperatures = setup_data_random_sampling["temperatures"]
    # qaic_temperatures = deepcopy(setup_data_random_sampling["temperatures"])
    random_numbers = setup_data_random_sampling["random_numbers"]
    qaic_random_numbers = deepcopy(setup_data_random_sampling["random_numbers"])
    
    # ---Run on CPU---
    qeff_start_time = perf_counter()
    qeff_output = sampler_forward(
        None,
        logits.to(torch.float16),
        # temperatures.to(torch.float16),
        random_numbers.to(torch.float16),
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
    onnx_path = "./on_device_sampling_onnx/test_sampler_random_sampling_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            qaic_logits,
            # qaic_temperatures,
            qaic_random_numbers,
        ),
        onnx_path,
        input_names=[
            "input_logits",
            # "temperatures",
            "random_numbers",
        ],
        output_names=[
            "logits", "probs", "next_tokens",
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
        # "temperatures": qaic_temperatures.detach().cpu().numpy(),
        "random_numbers": qaic_random_numbers.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    qaic_start_time = perf_counter()
    outputs = session.run(inputs)
    qaic_end_time = perf_counter()
    print("\nQAIC Output\n", outputs)
    print(f"Time Taken {(qaic_end_time - qaic_start_time) * 1000: .5f} ms\n")
    
    for k, v in outputs.items():
        print(k, "\n", torch.from_numpy(v), "\n")

    hw_output_logits = torch.from_numpy(outputs["logits"])
    hw_output_probs = torch.from_numpy(outputs["probs"])
    hw_output_next_tokens = torch.from_numpy(outputs["next_tokens"])

    # Compare outputs
    has_failed = False
    if not torch.allclose(qeff_output[0].to(torch.float32), hw_output_logits, atol=1e-4):
        print_difference_in_tensors(
            qeff_output[0].to(torch.float32),
            "Logits",
            hw_output_logits,
            "QAIC Logits",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(qeff_output[1].to(torch.float32), hw_output_probs, atol=1e-4):
        print_difference_in_tensors(
            qeff_output[1].to(torch.float32),
            "Probs",
            hw_output_probs,
            "QAIC Probs",
            1e-4,
        )
        has_failed = True

    # Summary Stats (Mean, Variance, ...)
    qeff_sumarry_stats = get_summary_statistics(qeff_output[2].squeeze(2).squeeze(1))
    vllm_summary_stats = get_summary_statistics(hw_output_next_tokens.squeeze(2).squeeze(1))

    for k in qeff_sumarry_stats.keys():
        print("\n")
        print(f"Next Tokens {k}: {qeff_sumarry_stats[k]}")
        print(f"vLLM Next Tokens {k}: {vllm_summary_stats[k]}")

        if not torch.allclose(qeff_sumarry_stats[k], vllm_summary_stats[k], atol=1):
            print_difference_in_tensors(
                qeff_sumarry_stats[k],
                f"Next Tokens {k}",
                vllm_summary_stats[k],
                f"vLLM Next Tokens {k}",
                0.1,
            )
            has_failed = True

    # KL Divergence
    kl_divergence = get_kl_divergence(
        qeff_output[2].squeeze(2).squeeze(1), hw_output_next_tokens.squeeze(2).squeeze(1), logits.shape[-1]
    )
    print("\n\nKL Divergence", kl_divergence)
    if kl_divergence > 0.1:
        has_failed = True

    # Z Test
    z_score, p_value = get_z_score(
        qeff_output[2].squeeze(2).squeeze(1),
        hw_output_next_tokens.squeeze(2).squeeze(1),
        qeff_output[2].shape[0],
        hw_output_next_tokens.shape[0],
    )
    print("\n\nZ Score", z_score)
    print("P Value", p_value)
    if p_value < 0.05:
        has_failed = True

    assert has_failed == False, "Test Failed"
    

def test_gpu_vs_qaic(setup_data_random_sampling):
    print(setup_data_random_sampling["seed"])

    logits = setup_data_random_sampling["logits"].cuda()
    qaic_logits = deepcopy(setup_data_random_sampling["logits"])

    # temperatures = setup_data_random_sampling["temperatures"].cuda()
    # qaic_temperatures = deepcopy(setup_data_random_sampling["temperatures"])
    random_numbers = setup_data_random_sampling["random_numbers"].cuda()
    qaic_random_numbers = deepcopy(setup_data_random_sampling["random_numbers"])
    
    # ---Run on CPU---
    qeff_start_time = perf_counter()
    qeff_output = sampler_forward(
        None,
        logits.to(torch.float16),
        # temperatures.to(torch.float16),
        random_numbers.to(torch.float16),
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
    onnx_path = "./on_device_sampling_onnx/test_sampler_random_sampling_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            qaic_logits,
            # qaic_temperatures,
            qaic_random_numbers,
        ),
        onnx_path,
        input_names=[
            "input_logits",
            # "temperatures",
            "random_numbers",
        ],
        output_names=[
            "logits", "probs", "next_tokens",
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
        # "temperatures": qaic_temperatures.detach().cpu().numpy(),
        "random_numbers": qaic_random_numbers.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    qaic_start_time = perf_counter()
    outputs = session.run(inputs)
    qaic_end_time = perf_counter()
    print("\nQAIC Output\n", outputs)
    print(f"Time Taken {(qaic_end_time - qaic_start_time) * 1000: .5f} ms\n")
    
    for k, v in outputs.items():
        print(k, "\n", torch.from_numpy(v), "\n")

    hw_output_logits = torch.from_numpy(outputs["logits"])
    hw_output_probs = torch.from_numpy(outputs["probs"])
    hw_output_next_tokens = torch.from_numpy(outputs["next_tokens"])

    # Compare outputs
    has_failed = False
    if not torch.allclose(qeff_output[0].cpu().to(torch.float32), hw_output_logits, atol=1e-4):
        print_difference_in_tensors(
            qeff_output[0].cpu().to(torch.float32),
            "Logits",
            hw_output_logits,
            "QAIC Logits",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(qeff_output[1].cpu().to(torch.float32), hw_output_probs, atol=1e-4):
        print_difference_in_tensors(
            qeff_output[1].cpu().to(torch.float32),
            "Probs",
            hw_output_probs,
            "QAIC Probs",
            1e-4,
        )
        has_failed = True

    # Summary Stats (Mean, Variance, ...)
    qeff_sumarry_stats = get_summary_statistics(qeff_output[2].squeeze(2).squeeze(1).cpu())
    vllm_summary_stats = get_summary_statistics(hw_output_next_tokens.squeeze(2).squeeze(1))

    for k in qeff_sumarry_stats.keys():
        print("\n")
        print(f"Next Tokens {k}: {qeff_sumarry_stats[k]}")
        print(f"vLLM Next Tokens {k}: {vllm_summary_stats[k]}")

        if not torch.allclose(qeff_sumarry_stats[k], vllm_summary_stats[k], atol=1):
            print_difference_in_tensors(
                qeff_sumarry_stats[k],
                f"Next Tokens {k}",
                vllm_summary_stats[k],
                f"vLLM Next Tokens {k}",
                0.1,
            )
            has_failed = True

    # KL Divergence
    kl_divergence = get_kl_divergence(
        qeff_output[2].squeeze(2).squeeze(1).cpu(), hw_output_next_tokens.squeeze(2).squeeze(1), logits.shape[-1]
    )
    print("\n\nKL Divergence", kl_divergence)
    if kl_divergence > 0.1:
        has_failed = True

    # Z Test
    z_score, p_value = get_z_score(
        qeff_output[2].squeeze(2).squeeze(1).cpu(),
        hw_output_next_tokens.squeeze(2).squeeze(1),
        qeff_output[2].shape[0],
        hw_output_next_tokens.shape[0],
    )
    print("\n\nZ Score", z_score)
    print("P Value", p_value)
    if p_value < 0.05:
        has_failed = True

    assert has_failed == False, "Test Failed"


def test_gpu_vs_vllm_gpu(setup_data_random_sampling):
    print(setup_data_random_sampling["seed"])

    logits = setup_data_random_sampling["logits"].cuda()
    vllm_logits = deepcopy(setup_data_random_sampling["logits"]).squeeze(1).cuda()

    batch_size = logits.shape[0]
    pseudo_random_generator = torch.Generator(device='cuda')
    generators = {
        i: pseudo_random_generator for i in range(batch_size)
    }
    random_numbers = setup_data_random_sampling["random_numbers"].cuda()
        
    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=None,
        no_top_p=True,
        no_top_k=True,
        no_min_p=True,
        generators=generators,
        max_num_logprobs=0,
        no_penalties=True,
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
        # temperatures,
        random_numbers,
    )
    vllm_output = vllm_sampler(vllm_logits, sampling_metadata)

    # print("\nLogits\n", qeff_output[0])
    # print("\nvLLM Logits\n", vllm_output[0])

    # print("\nProbs\n", qeff_output[1])
    # print("\nvLLM Probs\n", vllm_output[1])

    # print("\nNext Tokens\n", qeff_output[2])
    # print("\nvLLM Next Tokens\n", vllm_output[2])

    # Compare outputs
    has_failed = False
    if not torch.allclose(qeff_output[0].squeeze(1).cpu(), vllm_output[0].cpu(), atol=1e-4):
        print_difference_in_tensors(
            qeff_output[0].squeeze(1).cpu(),
            "Logits",
            vllm_output[0].cpu(),
            "vLLM Logits",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(qeff_output[1].squeeze(1).cpu(), vllm_output[1].cpu(), atol=1e-4):
        print_difference_in_tensors(
            qeff_output[1].squeeze(1).cpu(),
            "Probs",
            vllm_output[1].cpu(),
            "vLLM Probs",
            1e-4,
        )
        has_failed = True

    # Summary Stats (Mean, Variance, ...)
    qeff_sumarry_stats = get_summary_statistics(qeff_output[2].squeeze(2).squeeze(1).cpu())
    vllm_summary_stats = get_summary_statistics(vllm_output[2].cpu())

    for k in qeff_sumarry_stats.keys():
        print("\n")
        print(f"Next Tokens {k}: {qeff_sumarry_stats[k]}")
        print(f"vLLM Next Tokens {k}: {vllm_summary_stats[k]}")

        if not torch.allclose(qeff_sumarry_stats[k], vllm_summary_stats[k], atol=1):
            print_difference_in_tensors(
                qeff_sumarry_stats[k],
                f"Next Tokens {k}",
                vllm_summary_stats[k],
                f"vLLM Next Tokens {k}",
                0.1,
            )
            has_failed = True

    # KL Divergence
    kl_divergence = get_kl_divergence(
        qeff_output[2].squeeze(2).squeeze(1).cpu(), vllm_output[2].cpu(), logits.shape[-1]
    )
    print("\n\nKL Divergence", kl_divergence)
    if kl_divergence > 0.1:
        has_failed = True

    # Z Test
    z_score, p_value = get_z_score(
        qeff_output[2].squeeze(2).squeeze(1).cpu(),
        vllm_output[2].cpu(),
        qeff_output[2].shape[0],
        vllm_output[2].shape[0],
    )
    print("\n\nZ Score", z_score)
    print("P Value", p_value)
    if p_value < 0.05:
        has_failed = True

    assert has_failed == False, "Test Failed"
