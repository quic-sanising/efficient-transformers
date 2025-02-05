from dataclasses import dataclass
import torch

from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from typing import Optional, Tuple, Union

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
    top_ps: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    
    # print("-"*100)
    # params = locals()
    # for param, value in params.items():
    #     print(f"{param}: {value}")
    # print("-"*100)

    # Perform Sampling
    device = input_logits.device
    batch_size, spec_length, vocab_size = input_logits.shape
    logits = input_logits.reshape(batch_size * spec_length, vocab_size)  # Reshape tensor to 2D

    # Top K
    topk_values_asc, topk_indices_asc = torch.topk(logits, k=vocab_size, dim=1, largest=False)  # (batch_size * spec_length, vocab_size)

    # True values in this mask indicate the positions of the non-top K values
    topk_mask = torch.arange(topk_values_asc.shape[1], device=device).unsqueeze(0) < (topk_values_asc.size(1) - top_ks.to(torch.long)).unsqueeze(1).repeat(spec_length, 1)
    topk_values_asc[topk_mask] = torch.finfo(torch.float16).min

    # Top P
    top_probs = torch.softmax(topk_values_asc, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_probs_sum = torch.cumsum(top_probs, dim=1)
    top_p_mask = topk_probs_sum <= 1 - top_ps.unsqueeze(1).repeat(spec_length, 1) 
    top_p_mask[:, -1] = False
    topk_values_asc[top_p_mask] = torch.finfo(torch.float16).min

    logits = logits.scatter(1, topk_indices_asc, topk_values_asc)  # (batch_size * spec_length, vocab_size)
    logits = logits.reshape(batch_size, spec_length, vocab_size)

    return QEffCausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        repetition_penalty_retain_state=None,
        presence_penalty_retain_state=None,
    )
