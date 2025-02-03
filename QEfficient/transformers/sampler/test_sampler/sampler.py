from dataclasses import dataclass
import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union

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
    input_ids: torch.LongTensor,
    input_logits: torch.Tensor,
    last_accepted_output_tokens: Optional[torch.Tensor] = None,  # (batch_size, spec_length or less)
    repetition_penalty_retain_state: Optional[torch.Tensor] = None,
    repetition_penalties: Optional[torch.Tensor] = None,
    presence_penalty_retain_state: Optional[torch.Tensor] = None,
    presence_penalties: Optional[torch.Tensor] = None,
    temperatures: Optional[torch.Tensor] = None,
    top_ks: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # Perform Sampling
    batch_size, spec_length, vocab_size = input_logits.shape
    logits = input_logits.reshape(batch_size * spec_length, vocab_size)  # Reshape tensor to 2D

    if input_ids.shape[1] != spec_length:  # Prefill phase, initialize retained states
        repetition_penalty_retain_state = torch.mul(repetition_penalty_retain_state, 0)
        presence_penalty_retain_state = torch.mul(presence_penalty_retain_state, 0)
        repetition_penalty_retain_state.scatter_(1, input_ids, 1)
    else:  # Decode phase, update retained states
        repetition_penalty_retain_state.scatter_(1, last_accepted_output_tokens, 1)
        presence_penalty_retain_state.scatter_(1, last_accepted_output_tokens, 1)
        # TODO: For frequency retain state, first gather and then scatter

    # Repetition Penalty
    if (repetition_penalties != 1.).any():
        repetition_penalties = repetition_penalties.unsqueeze(1).repeat(spec_length, vocab_size)  # (batch_size,) -> (batch_size * spec_length, vocab_size)
        repetition_penalty_retain_state = repetition_penalty_retain_state.repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        repetition_penalties[~repetition_penalty_retain_state.bool()] = 1.0
        logits = torch.where(
            logits > 0, logits / repetition_penalties, logits * repetition_penalties
        )

    # Presence Penalty
    if (presence_penalties != 0.).any():
        presence_penalties = presence_penalties.unsqueeze(1).repeat(spec_length, 1)  # (batch_size,) -> (batch_size * spec_length, 1)
        presence_penalty_retain_state = presence_penalty_retain_state.repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        logits -= presence_penalties * presence_penalty_retain_state

    # TODO: Frequency Penalty

    # Temperature Scaling
    if (temperatures != 0).any():
        temperatures = temperatures.unsqueeze(1).repeat(spec_length, 1)  # (batch_size,) -> (batch_size * spec_length, 1)
        logits = torch.where(temperatures != 0, logits / temperatures, logits)

    # Top K
    topk_values, topk_indices = torch.topk(logits, k=vocab_size, dim=1)  # (batch_size * spec_length, vocab_size)

    # True values in this mask indicate the positions of the top K values.
    topk_inverted_mask = torch.arange(topk_values.shape[1]).unsqueeze(0) < top_ks.unsqueeze(1).repeat(spec_length, 1)
    topk_values[~topk_inverted_mask] = -float("inf")

    logits.scatter_(1, topk_indices, topk_values)  # (batch_size * spec_length, vocab_size)

    # Softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size * spec_length, vocab_size)

    # Reshape tensor back to 3D
    logits = logits.reshape(batch_size, spec_length, vocab_size)
    probs = probs.reshape(batch_size, spec_length, vocab_size)
    repetition_penalty_retain_state = repetition_penalty_retain_state.reshape(spec_length, batch_size, vocab_size)[0]  # Undo spec_length repetition
    presence_penalty_retain_state = presence_penalty_retain_state.reshape(spec_length, batch_size, vocab_size)[0]

    return QEffCausalLMOutputWithPast(
        loss=None,
        logits=probs,  # Return probabilities or sampled next tokens instead of logits
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        repetition_penalty_retain_state=repetition_penalty_retain_state,
        presence_penalty_retain_state=presence_penalty_retain_state,
    )
