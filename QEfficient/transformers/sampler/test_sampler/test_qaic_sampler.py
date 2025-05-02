from copy import deepcopy
from time import perf_counter
import json
import subprocess
import torch
import torch.nn as nn

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.sampler.test_sampler.sampler import sampler_forward
from QEfficient.transformers.sampler.test_sampler.make_inputs import write_io_files
from QEfficient.transformers.sampler.test_sampler.utils import print_difference_in_tensors
from QEfficient.transformers.sampler.test_sampler.vllm_sampler import (
    Sampler,
    SamplingMetadata,
)
from QEfficient.utils import constants


def test_cpu_vs_vllm_cpu(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"]
    output_token_ids = setup_data["output_token_ids"]

    logits = setup_data["logits"]
    vllm_logits = deepcopy(setup_data["logits"]).squeeze(1)
    
    position_ids = setup_data["position_ids"]
    batch_index = setup_data["batch_index"]

    past_repetition_penalty_buffer = setup_data["past_repetition_penalty_buffer"]
    past_presence_penalty_buffer = setup_data["past_presence_penalty_buffer"]

    repetition_penalties = setup_data["repetition_penalties"]
    presence_penalties = setup_data["presence_penalties"]

    temperatures = setup_data["temperatures"]

    top_ks = setup_data["top_ks"]
    top_ps = setup_data["top_ps"]
    min_ps = setup_data["min_ps"]

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=temperatures.squeeze(1),
        all_greedy=False,
        all_random=False,
        top_p=top_ps.squeeze(1),
        top_k=top_ks.squeeze(1),
        min_p=min_ps.squeeze(1),
        no_top_p=False,
        no_top_k=False,
        no_min_p=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=None,
        presence_penalties=presence_penalties.squeeze(1),
        repetition_penalties=repetition_penalties.squeeze(1),
        output_token_ids=output_token_ids.tolist(),
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        position_ids,
        batch_index,
        logits,
        output_token_ids[:, -1:],
        past_repetition_penalty_buffer,
        repetition_penalties,
        past_presence_penalty_buffer,
        presence_penalties,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
    )
    vllm_output_probs, vllm_prompt_mask, vllm_output_mask = vllm_sampler(
        vllm_logits, sampling_metadata
    )

    # print(qeff_output.probs.squeeze(1))
    # print(vllm_output_probs)

    # Compare outputs
    has_failed = False
    if not torch.allclose(
        qeff_output.probs.squeeze(1), vllm_output_probs, atol=1e-3
    ): 
        print_difference_in_tensors(
            qeff_output.probs.squeeze(1),
            "Probs",
            vllm_output_probs,
            "vLLM Probs",
            1e-3,
        )
        has_failed = True

    if not torch.allclose(
        qeff_output.past_repetition_penalty_buffer, vllm_prompt_mask | vllm_output_mask
    ):
        print_difference_in_tensors(
            qeff_output.past_repetition_penalty_buffer, 
            "Past Repetition Penalty Buffer",
            vllm_prompt_mask | vllm_output_mask,
            "vLLM Past Repetition Penalty Buffer",
            1e-3,
        )
        has_failed = True
        
    if not torch.allclose(
        qeff_output.past_presence_penalty_buffer, vllm_output_mask
    ): 
        print_difference_in_tensors(
            qeff_output.past_presence_penalty_buffer,
            "Past Presence Penalty Buffer",
            vllm_output_mask,
            "vLLM Past Presence Penalty Buffer",
            1e-3,
        )
        has_failed = True
    
    assert not has_failed, "Test failed"


def test_cpu_vs_qaic(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"]
    output_token_ids = setup_data["output_token_ids"]

    logits = setup_data["logits"]
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data["logits"])
    print("QAIC Input Logits", qaic_logits)

    position_ids = setup_data["position_ids"]
    batch_index = setup_data["batch_index"]

    past_repetition_penalty_buffer = setup_data["past_repetition_penalty_buffer"]
    qaic_past_repetition_penalty_buffer = deepcopy(setup_data["past_repetition_penalty_buffer"])
    past_presence_penalty_buffer = setup_data["past_presence_penalty_buffer"]
    qaic_past_presence_penalty_buffer = deepcopy(setup_data["past_presence_penalty_buffer"])

    repetition_penalties = setup_data["repetition_penalties"]
    presence_penalties = setup_data["presence_penalties"]

    temperatures = setup_data["temperatures"]

    top_ks = setup_data["top_ks"]
    top_ps = setup_data["top_ps"]
    min_ps = setup_data["min_ps"]

    # ---Run on CPU---
    qeff_start_time = perf_counter()
    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        position_ids,
        batch_index,
        logits.to(torch.float16),
        output_token_ids[:, -1:],
        past_repetition_penalty_buffer,
        repetition_penalties.to(torch.float16),
        past_presence_penalty_buffer,
        presence_penalties.to(torch.float16),
        temperatures.to(torch.float16),
        top_ks,
        top_ps,
        min_ps,
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
    onnx_path = "./on_device_sampling_onnx/test_sampler_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            output_token_ids[:, -1:],
            position_ids,
            batch_index,
            qaic_logits,
            output_token_ids[:, -1:],
            qaic_past_repetition_penalty_buffer,
            repetition_penalties,
            qaic_past_presence_penalty_buffer,
            presence_penalties,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
        ),
        onnx_path,
        input_names=[
            "input_ids",
            "position_ids",
            "batch_index",
            "input_logits",
            "last_accepted_output_tokens",
            "past_repetition_penalty_buffer",
            "repetition_penalties",
            "past_presence_penalty_buffer",
            "presence_penalties",
            "temperatures",
            "top_ks",
            "top_ps",
            "min_ps",
        ],
        output_names=[
            "probs",
            # "next_tokens",
            "past_repetition_penalty_buffer_RetainedState",
            "past_presence_penalty_buffer_RetainedState",
        ],
        dynamo=False,
        verbose=True,
        opset_version=constants.ONNX_EXPORT_OPSET,
    )

    # Compile QPC file
    qpc_dir_path = "./on_device_sampling_qpcs/"
    compile_cmd = [
        "/opt/qti-aic/exec/qaic-exec",
        "-aic-hw",
        "-aic-hw-version=2.0",
        "-stats-level=70",
        f"-m={onnx_path}",
        "-compile-only",
        "-retained-state",
        "-convert-to-fp16",
        "-aic-num-cores=16",
        f"-aic-binary-dir={qpc_dir_path}",
        "-mxfp6-matmul",
    ]
    # Write mdp_config.json file
    mdp_ts_num_devices = setup_data.get("num_devices", 1)
    if mdp_ts_num_devices > 1:
        mdp_ts_json = f"./mdp_ts_{mdp_ts_num_devices}.json"
        with open(mdp_ts_json, "w") as fp:
            json.dump(
                {
                    "connections": [{"devices": list(range(mdp_ts_num_devices)), "type": "p2p"}],
                    "partitions": [
                        {
                            "name": "Partition0",
                            "devices": [{"deviceId": d, "numCores": 16} for d in range(mdp_ts_num_devices)],
                        }
                    ],
                },
                fp,
                indent=4,
            )
        compile_cmd.append(f"-mdp-load-partition-config={mdp_ts_json}")
    
    subprocess.run(["rm", "-rf", f"{qpc_dir_path}"])
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    # print(result.stdout)
    if (result.returncode != 0):
        print(result.stderr)

    # Run QPC file
    session = QAICInferenceSession(qpc_path=qpc_dir_path, device_ids=list(range(11, 11 + mdp_ts_num_devices)), enable_debug_logs=False)
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "position_ids": position_ids.detach().cpu().numpy(),
        "batch_index": batch_index.detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "last_accepted_output_tokens": output_token_ids[:, -1:].detach().cpu().numpy(),
        "past_repetition_penalty_buffer": qaic_past_repetition_penalty_buffer.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "past_presence_penalty_buffer": qaic_past_presence_penalty_buffer.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
        "min_ps": min_ps.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    
    write_io_files(inputs, dict(), "./pytest_outputs/", f"decode", "aic_batch_io", True)
    
    qaic_start_time = perf_counter()
    outputs = session.run(inputs)
    qaic_end_time = perf_counter()
    print("\nQAIC Output\n", outputs)
    print(f"Time Taken {(qaic_end_time - qaic_start_time) * 1000: .5f} ms\n")

    hw_output_probs = torch.from_numpy(outputs["probs"])
    hw_past_repetition_penalty_buffer = torch.from_numpy(outputs["past_repetition_penalty_buffer_RetainedState"])
    hw_past_presence_penalty_buffer = torch.from_numpy(outputs["past_presence_penalty_buffer_RetainedState"])

    print("\Probs\n", qeff_output.probs)
    print("\nQAIC Probs\n", hw_output_probs)

    print("\nPast Repetition Penalty Buffer\n", qeff_output.past_repetition_penalty_buffer)
    print("\nQAIC Past Repetition Penalty Buffer\n", hw_past_repetition_penalty_buffer)

    print("\nPast Presence Penalty Buffer\n", qeff_output.past_presence_penalty_buffer)
    print("\nQAIC Past Presence Penalty Buffer\n", hw_past_presence_penalty_buffer)

    # Compare outputs
    has_failed = False
    if not torch.allclose(
        qeff_output.probs.to(torch.float32), hw_output_probs, atol=2**-10
    ): 
        print_difference_in_tensors(
            qeff_output.probs.to(torch.float32),
            "Probs",
            hw_output_probs,
            "QAIC Probs",
            2**-10,
        )
        has_failed = True

    if not torch.allclose(
        qeff_output.past_repetition_penalty_buffer.to(torch.int8), hw_past_repetition_penalty_buffer
    ):
        print_difference_in_tensors(
            qeff_output.past_repetition_penalty_buffer.to(torch.int8), 
            "Past Repetition Penalty Buffer",
            hw_past_repetition_penalty_buffer,
            "QAIC Past Repetition Penalty Buffer",
        )
        has_failed = True
        
    if not torch.allclose(
        qeff_output.past_presence_penalty_buffer.to(torch.int8), hw_past_presence_penalty_buffer
    ): 
        print_difference_in_tensors(
            qeff_output.past_presence_penalty_buffer.to(torch.int8),
            "Past Presence Penalty Buffer",
            hw_past_presence_penalty_buffer,
            "QAIC Past Presence Penalty Buffer",
        )
        has_failed = True
    
    assert not has_failed, "Test failed"


def test_gpu_vs_qaic(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"].cuda()
    output_token_ids = setup_data["output_token_ids"].cuda()

    logits = setup_data["logits"].cuda()
    print("Input Logits", logits)
    qaic_logits = deepcopy(setup_data["logits"])
    print("QAIC Input Logits", qaic_logits)

    position_ids = setup_data["position_ids"].cuda()
    batch_index = setup_data["batch_index"].cuda()

    past_repetition_penalty_buffer = setup_data["past_repetition_penalty_buffer"].cuda()
    qaic_past_repetition_penalty_buffer = deepcopy(setup_data["past_repetition_penalty_buffer"])
    past_presence_penalty_buffer = setup_data["past_presence_penalty_buffer"].cuda()
    qaic_past_presence_penalty_buffer = deepcopy(setup_data["past_presence_penalty_buffer"])

    repetition_penalties = setup_data["repetition_penalties"].cuda()
    presence_penalties = setup_data["presence_penalties"].cuda()

    temperatures = setup_data["temperatures"].cuda()

    top_ks = setup_data["top_ks"].cuda()
    top_ps = setup_data["top_ps"].cuda()
    min_ps = setup_data["min_ps"].cuda()

    # ---Run on GPU---
    qeff_start_time = perf_counter()
    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        position_ids,
        batch_index,
        logits.to(torch.float16),
        output_token_ids[:, -1:],
        past_repetition_penalty_buffer,
        repetition_penalties.to(torch.float16),
        past_presence_penalty_buffer,
        presence_penalties.to(torch.float16),
        temperatures.to(torch.float16),
        top_ks,
        top_ps,
        min_ps,
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
    onnx_path = "./on_device_sampling_onnx/test_sampler_hardware.onnx"
    subprocess.run(["rm", "-rf", f"{onnx_path}"])
    torch.onnx.export(
        model,
        (
            None,
            output_token_ids[:, -1:],
            position_ids,
            batch_index,
            qaic_logits,
            output_token_ids[:, -1:],
            qaic_past_repetition_penalty_buffer,
            repetition_penalties,
            qaic_past_presence_penalty_buffer,
            presence_penalties,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
        ),
        onnx_path,
        input_names=[
            "input_ids",
            "position_ids",
            "batch_index",
            "input_logits",
            "last_accepted_output_tokens",
            "past_repetition_penalty_buffer",
            "repetition_penalties",
            "past_presence_penalty_buffer",
            "presence_penalties",
            "temperatures",
            "top_ks",
            "top_ps",
            "min_ps",
        ],
        output_names=[
            "probs",
            # "next_tokens",
            "past_repetition_penalty_buffer_RetainedState",
            "past_presence_penalty_buffer_RetainedState",
        ],
        dynamo=False,
        verbose=False,
    )

    # Compile QPC file
    qpc_dir_path = "./on_device_sampling_qpcs/"
    compile_cmd = [
        "/opt/qti-aic/exec/qaic-exec",
        "-aic-hw",
        "-aic-hw-version=2.0",
        "-stats-level=70",
        f"-m={onnx_path}",
        "-compile-only",
        "-retained-state",
        "-convert-to-fp16",
        "-aic-num-cores=16",
        f"-aic-binary-dir={qpc_dir_path}",
        "-mxfp6-matmul",
    ]
    subprocess.run(["rm", "-rf", f"{qpc_dir_path}"])
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    # print(result.stdout)
    if (result.returncode != 0):
        print(result.stderr)

    # Run QPC file
    session = QAICInferenceSession(qpc_path=qpc_dir_path, device_ids=[10], enable_debug_logs=False)
    inputs = {
        # "input_ids": output_token_ids[:, -1:].detach().cpu().numpy(),
        "position_ids": position_ids.detach().cpu().numpy(),
        "batch_index": batch_index.detach().cpu().numpy(),
        "input_logits": qaic_logits.detach().cpu().numpy(),
        "last_accepted_output_tokens": output_token_ids[:, -1:].detach().cpu().numpy(),
        "past_repetition_penalty_buffer": qaic_past_repetition_penalty_buffer.detach().cpu().numpy(),
        "repetition_penalties": repetition_penalties.detach().cpu().numpy(),
        "past_presence_penalty_buffer": qaic_past_presence_penalty_buffer.detach().cpu().numpy(),
        "presence_penalties": presence_penalties.detach().cpu().numpy(),
        "temperatures": temperatures.detach().cpu().numpy(),
        "top_ks": top_ks.detach().cpu().numpy(),
        "top_ps": top_ps.detach().cpu().numpy(),
        "min_ps": min_ps.detach().cpu().numpy(),
    }
    print("\nQAIC Input\n", inputs)
    qaic_start_time = perf_counter()
    outputs = session.run(inputs)
    qaic_end_time = perf_counter()
    print("\nQAIC Output\n", outputs)
    print(f"Time Taken {(qaic_end_time - qaic_start_time) * 1000: .5f} ms\n")

    hw_output_probs = torch.from_numpy(outputs["probs"])
    hw_past_repetition_penalty_buffer = torch.from_numpy(outputs["past_repetition_penalty_buffer_RetainedState"])
    hw_past_presence_penalty_buffer = torch.from_numpy(outputs["past_presence_penalty_buffer_RetainedState"])

    print("\Probs\n", qeff_output.probs)
    print("\nQAIC Probs\n", hw_output_probs)

    print("\nPast Repetition Penalty Buffer\n", qeff_output.past_repetition_penalty_buffer)
    print("\nQAIC Past Repetition Penalty Buffer\n", hw_past_repetition_penalty_buffer)

    print("\nPast Presence Penalty Buffer\n", qeff_output.past_presence_penalty_buffer)
    print("\nQAIC Past Presence Penalty Buffer\n", hw_past_presence_penalty_buffer)

    # Compare outputs
    has_failed = False
    if not torch.allclose(
        qeff_output.probs.cpu().to(torch.float32), hw_output_probs, atol=1e-4
    ): 
        print_difference_in_tensors(
            qeff_output.probs.cpu().to(torch.float32),
            "Probs",
            hw_output_probs,
            "QAIC Probs",
            1e-4,
        )
        has_failed = True

    if not torch.allclose(
        qeff_output.past_repetition_penalty_buffer.cpu().to(torch.int8), hw_past_repetition_penalty_buffer
    ):
        print_difference_in_tensors(
            qeff_output.past_repetition_penalty_buffer.cpu().to(torch.int8), 
            "Past Repetition Penalty Buffer",
            hw_past_repetition_penalty_buffer,
            "QAIC Past Repetition Penalty Buffer",
        )
        has_failed = True
        
    if not torch.allclose(
        qeff_output.past_presence_penalty_buffer.cpu().to(torch.int8), hw_past_presence_penalty_buffer
    ): 
        print_difference_in_tensors(
            qeff_output.past_presence_penalty_buffer.cpu().to(torch.int8),
            "Past Presence Penalty Buffer",
            hw_past_presence_penalty_buffer,
            "QAIC Past Presence Penalty Buffer",
        )
        has_failed = True
    
    assert not has_failed, "Test failed"


def test_gpu_vs_vllm_gpu(setup_data):
    print(setup_data["seed"])

    prompt_token_ids = setup_data["prompt_token_ids"].cuda()
    output_token_ids = setup_data["output_token_ids"].cuda()

    logits = setup_data["logits"].cuda()
    vllm_logits = deepcopy(setup_data["logits"]).squeeze(1).cuda()
    
    position_ids = setup_data["position_ids"].cuda()
    batch_index = setup_data["batch_index"].cuda()

    past_repetition_penalty_buffer = setup_data["past_repetition_penalty_buffer"].cuda()
    past_presence_penalty_buffer = setup_data["past_presence_penalty_buffer"].cuda()

    repetition_penalties = setup_data["repetition_penalties"].cuda()
    presence_penalties = setup_data["presence_penalties"].cuda()

    temperatures = setup_data["temperatures"].cuda()

    top_ks = setup_data["top_ks"].cuda()
    top_ps = setup_data["top_ps"].cuda()
    min_ps = setup_data["min_ps"].cuda()

    vllm_sampler = Sampler()
    sampling_metadata = SamplingMetadata(
        temperature=temperatures.squeeze(1),
        all_greedy=False,
        all_random=False,
        top_p=top_ps.squeeze(1),
        top_k=top_ks.squeeze(1),
        min_p=min_ps.squeeze(1),
        no_top_p=False,
        no_top_k=False,
        no_min_p=False,
        generators=None,
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=None,
        presence_penalties=presence_penalties.squeeze(1),
        repetition_penalties=repetition_penalties.squeeze(1),
        output_token_ids=output_token_ids.tolist(),
        min_tokens=None,
        stop_token_ids=None,
    )

    qeff_output = sampler_forward(
        None,
        output_token_ids[:, -1:],
        position_ids,
        batch_index,
        logits,
        output_token_ids[:, -1:],
        past_repetition_penalty_buffer,
        repetition_penalties,
        past_presence_penalty_buffer,
        presence_penalties,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
    )
    vllm_output_probs, vllm_prompt_mask, vllm_output_mask = vllm_sampler(
        vllm_logits, sampling_metadata
    )

    print(qeff_output.probs.squeeze(1))
    print(vllm_output_probs)

    # Compare outputs
    has_failed = False
    if not torch.allclose(
        qeff_output.probs.squeeze(1).cpu(), vllm_output_probs.cpu(), atol=1e-3
    ): 
        print_difference_in_tensors(
            qeff_output.probs.squeeze(1).cpu(),
            "Probs",
            vllm_output_probs.cpu(),
            "vLLM Probs",
            1e-3,
        )
        has_failed = True

    if not torch.allclose(
        qeff_output.past_repetition_penalty_buffer.cpu(), vllm_prompt_mask.cpu() | vllm_output_mask.cpu()
    ):
        print_difference_in_tensors(
            qeff_output.past_repetition_penalty_buffer.cpu(), 
            "Past Repetition Penalty Buffer",
            vllm_prompt_mask.cpu() | vllm_output_mask.cpu(),
            "vLLM Past Repetition Penalty Buffer",
            1e-3,
        )
        has_failed = True
        
    if not torch.allclose(
        qeff_output.past_presence_penalty_buffer.cpu(), vllm_output_mask.cpu()
    ): 
        print_difference_in_tensors(
            qeff_output.past_presence_penalty_buffer.cpu(),
            "Past Presence Penalty Buffer",
            vllm_output_mask.cpu(),
            "vLLM Past Presence Penalty Buffer",
            1e-3,
        )
        has_failed = True
    
    assert not has_failed, "Test failed"
