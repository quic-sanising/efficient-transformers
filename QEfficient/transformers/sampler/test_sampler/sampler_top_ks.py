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
