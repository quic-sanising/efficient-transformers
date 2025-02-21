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
