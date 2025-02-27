import torch

from typing import Optional


def sampler_forward(
    self,
    input_logits: torch.Tensor,
    # temperatures: Optional[torch.Tensor] = None,
    random_numbers: Optional[torch.Tensor] = None,
):
    
    # print("-"*100)
    # params = locals()
    # for param, value in params.items():
    #     print(f"{param}: {value}")
    # print("-"*100)
    
    # Move to device
    device = input_logits.device
    # temperatures = temperatures.to(device)

    # Perform Sampling
    batch_size, spec_length, vocab_size = input_logits.shape
    logits = input_logits.reshape(batch_size * spec_length, vocab_size).to(device)  # Reshape tensor to 2D

    # Softmax
    probs = torch.softmax(logits, dim=1)  # (batch_size * spec_length, vocab_size)

    # Sample the next tokens
    # greedy_samples = torch.argmax(probs, dim=-1, keepdim=True)  # Greedy Sampling
    gumbel_noise = -torch.log(-torch.log(random_numbers.unsqueeze(1).repeat(spec_length, 1)))  # Gumbel-Max Trick
    y = probs + gumbel_noise
    random_samples = torch.argmax(y, dim=-1, keepdim=True)  # Random Sampling
    # next_tokens = torch.where(temperatures == 0, greedy_samples, random_samples)  # (batch_size * spec_length, 1)

    # Reshape tensor back to 3D
    logits = logits.reshape(batch_size, spec_length, vocab_size)
    probs = probs.reshape(batch_size, spec_length, vocab_size)
    next_tokens = random_samples.reshape(batch_size, spec_length, 1)

    return logits, probs, next_tokens
