import copy
import torch

from QEfficient.transformers.sampler.test_sampler.sampler_top_ps import sampler_forward
from QEfficient.transformers.sampler.test_sampler.vllm_sampler_topkp import (
    Sampler,
    SamplingMetadata,
)


def test_cpu_vs_vllm_cpu(setup_data_top_ps):
    print(setup_data_top_ps["seed"])

    logits = setup_data_top_ps["logits"]

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
        copy.deepcopy(logits),
        top_ks,
        top_ps,
    )
    vllm_output_logits = vllm_sampler(copy.deepcopy(logits).squeeze(1), sampling_metadata)
    print(f"QEff Output: {qeff_output.logits.squeeze(1)}")
    print(f"VLLM Output: {vllm_output_logits}")

    assert torch.allclose(
        qeff_output.logits.squeeze(1), vllm_output_logits, atol=1e-6
    ), "Output logits do not match"
