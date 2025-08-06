from dataclasses import dataclass
import torch

# from QEfficient.customop import CtxScatterFunc
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from typing import Optional, Tuple, Union

MAX_TOP_K_IDS = 512

@dataclass
class QEffCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    repetition_penalty_retain_state: Optional[torch.Tensor] = None
    presence_penalty_retain_state: Optional[torch.Tensor] = None


def sampler_forward(
    self,
    input_logits: torch.Tensor,
    top_ks: Optional[torch.Tensor] = None,
):
    
    # print("-"*100)
    # params = locals()
    # for param, value in params.items():
    #     print(f"{param}: {value}")
    # print("-"*100)

    # Perform Sampling
    device = input_logits.device
    batch_size, spec_length, vocab_size = input_logits.shape
    logits = input_logits.reshape(-1, vocab_size)  # Reshape tensor to 2D

    # Top K
    topk_values, topk_indices = torch.topk(logits, k=MAX_TOP_K_IDS, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_values_asc = topk_values.flip(dims=[1])
    topk_indices_asc = topk_indices.flip(dims=[1])
    top_ks[top_ks > MAX_TOP_K_IDS] = MAX_TOP_K_IDS  # Clip k to max value
    # True values in this mask indicate the positions of the non-top K values
    topk_mask = torch.arange(topk_values_asc.shape[1]).unsqueeze(0) < (topk_values_asc.size(1) - top_ks.to(torch.long)).repeat(spec_length, 1)  # (batch_size * spec_length, MAX_TOP_K_IDS)
    topk_values_asc[topk_mask] = torch.finfo(torch.float16).min

    # logits = CtxScatterFunc.apply(logits.unsqueeze(1), topk_indices.unsqueeze(1), topk_values.unsqueeze(1)).squeeze(1)
    logits.fill_(torch.finfo(torch.float16).min)
    logits = logits.scatter(1, topk_indices_asc, topk_values_asc)  # (batch_size * spec_length, vocab_size)
    logits = logits.reshape(-1, spec_length, vocab_size)

    return logits, topk_values, topk_indices, topk_mask
