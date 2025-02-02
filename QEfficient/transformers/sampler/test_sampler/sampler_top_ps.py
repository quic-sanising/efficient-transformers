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
    logits: torch.Tensor,
    top_ks: Optional[torch.Tensor] = None,
    top_ps: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    
    # print("-"*100)
    # params = locals()
    # for param, value in params.items():
    #     print(f"{param}: {value}")
    # print("-"*100)

    # Perform Sampling
    batch_size, spec_length, vocab_size = logits.shape
    logits = logits.reshape(batch_size * spec_length, vocab_size)  # Reshape tensor to 2D

    # Top K
    topk_values, topk_indices = torch.topk(logits, k=vocab_size, dim=1)  # (batch_size * spec_length, vocab_size)

    # True values in this mask indicate the positions of the top K values.
    topk_inverted_mask = torch.arange(topk_values.shape[1]).unsqueeze(0) < top_ks.unsqueeze(1).repeat(spec_length, 1)
    topk_values[~topk_inverted_mask] = -float("inf")

    # Top P
    top_probs = torch.softmax(topk_values, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_probs_sum = torch.cumsum(top_probs, dim=1)
    top_p_mask = topk_probs_sum > top_ps.unsqueeze(1).repeat(spec_length, 1) 

    # Shift the mask to the right by one position.
    shifted_mask = torch.zeros_like(top_p_mask)
    shifted_mask[:, 1:] = top_p_mask[:, :-1]

    # Ensure the first position is always False.
    shifted_mask[:, 0] = False
    # True values in
    # this mask indicate the positions where the cumulative probability exceeds the
    # threshold, and these values are set to 0.
    # top_probs = torch.where(shifted_mask, torch.tensor(0.0), top_probs)  # (batch_size * spec_length, vocab_size)

    # # Scatter the top probs into the probs tensor
    # probs = torch.zeros(logits.shape, dtype=torch.float)
    # probs.scatter_(1, topk_indices, top_probs)  # (batch_size * spec_length, vocab_size)

    # # Reshape tensor back to 3D
    # probs = probs.reshape(batch_size, spec_length, vocab_size)

    topk_values[top_p_mask] = -float("inf")
    logits.scatter_(1, topk_indices, topk_values)  # (batch_size * spec_length, vocab_size)
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
