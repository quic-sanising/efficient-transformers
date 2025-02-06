import pytest
import torch


@pytest.fixture(params=[8, 16, 64, 128])
def sequence_length(request):
    return request.param


@pytest.fixture(params=[1, 4, 8, 16, 32])
# @pytest.fixture(params=[1])
def batch_size(request):
    return request.param


@pytest.fixture(params=[10, 100, 1024, 2048, 4096])
# @pytest.fixture(params=[10])
def vocab_size(request):
    return request.param


@pytest.fixture(params=[128, 512, 4096, 8192])
def ctx_length(request, sequence_length):
    return max(request.param, sequence_length + 1)


@pytest.fixture
def setup_data_penalties(sequence_length, batch_size, vocab_size, ctx_length):
    import numpy as np

    seed = np.random.randint(1, 101)
    torch.manual_seed(seed)

    prompt_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    output_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, ctx_length))

    logits = torch.randn(batch_size, 1, vocab_size)

    repetition_penalty_retain_state = torch.zeros(batch_size, vocab_size, dtype=torch.int32)
    presence_penalty_retain_state = torch.zeros(batch_size, vocab_size, dtype=torch.int32)

    repetition_penalty_retain_state.scatter_(1, prompt_token_ids, 1)
    repetition_penalty_retain_state.scatter_(1, output_token_ids[:, :-1], 1)
    presence_penalty_retain_state.scatter_(1, output_token_ids[:, :-1], 1)

    repetition_penalties = torch.randint(1, 21, (batch_size,)) / 10.0
    presence_penalties = torch.randint(-10, 10, (batch_size,)) / 10.0

    return {
        "seed": seed,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "ctx_length": ctx_length,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "logits": logits,
        "repetition_penalty_retain_state": repetition_penalty_retain_state,
        "presence_penalty_retain_state": presence_penalty_retain_state,
        "repetition_penalties": repetition_penalties,
        "presence_penalties": presence_penalties,
    }


@pytest.fixture
def setup_data_top_ks(batch_size, vocab_size):
    import numpy as np

    seed = np.random.randint(1, 101)
    torch.manual_seed(seed)

    logits = torch.randn(batch_size, 1, vocab_size)
    top_ks = torch.randint(1, vocab_size, (batch_size,))  # Between 1 and vocab_size

    print("top_ks", top_ks)

    return {
        "seed": seed,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "logits": logits,
        "top_ks": top_ks,
    }


@pytest.fixture
def setup_data_top_ps(batch_size, vocab_size):
    import numpy as np

    seed = np.random.randint(1, 101)
    torch.manual_seed(seed)

    logits = torch.randn(batch_size, 1, vocab_size)
    top_ks = torch.randint(1, vocab_size, (batch_size,))  # Between 1 and vocab_size
    top_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99

    print("top_ps", top_ps)

    return {
        "seed": seed,
        "logits": logits,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "top_ks": top_ks,
        "top_ps": top_ps,
    }


@pytest.fixture
def setup_data_min_ps(batch_size, vocab_size):
    import numpy as np

    seed = np.random.randint(1, 101)
    torch.manual_seed(seed)

    logits = torch.randn(batch_size, 1, vocab_size)
    top_ks = torch.randint(1, vocab_size, (batch_size,))  # Between 1 and vocab_size
    top_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99
    min_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99

    print("min_ps", min_ps)

    return {
        "seed": seed,
        "logits": logits,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "top_ks": top_ks,
        "top_ps": top_ps,
        "min_ps": min_ps,
    }


@pytest.fixture
def setup_data(sequence_length, batch_size, vocab_size, ctx_length):
    import numpy as np

    seed = np.random.randint(1, 101)
    torch.manual_seed(seed)

    prompt_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    output_token_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, ctx_length))

    logits = torch.randn(batch_size, 1, vocab_size)

    repetition_penalty_retain_state = torch.zeros(batch_size, vocab_size, dtype=torch.int32)
    presence_penalty_retain_state = torch.zeros(batch_size, vocab_size, dtype=torch.int32)

    repetition_penalty_retain_state.scatter_(1, prompt_token_ids, 1)
    repetition_penalty_retain_state.scatter_(1, output_token_ids[:, :-1], 1)
    presence_penalty_retain_state.scatter_(1, output_token_ids[:, :-1], 1)

    repetition_penalties = torch.randint(1, 21, (batch_size,)) / 10.0
    presence_penalties = torch.randint(-10, 10, (batch_size,)) / 10.0

    temperatures = torch.randint(1, 11, (batch_size,)) / 10.0

    top_ks = torch.randint(1, vocab_size, (batch_size,))  # Between 1 and vocab_size
    top_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99


    return {
        "seed": seed,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "ctx_length": ctx_length,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "logits": logits,
        "repetition_penalty_retain_state": repetition_penalty_retain_state,
        "presence_penalty_retain_state": presence_penalty_retain_state,
        "repetition_penalties": repetition_penalties,
        "presence_penalties": presence_penalties,
        "temperatures": temperatures,
        "top_ks": top_ks,
        "top_ps": top_ps,
    }
