import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--sequence-length", type=int, default=None)
    parser.addoption("--batch-size", type=int, default=None)
    parser.addoption("--vocab-size", type=int, default=None)
    parser.addoption("--ctx-length", type=int, default=None)


def pytest_generate_tests(metafunc):
    if 'sequence_length' in metafunc.fixturenames:
        metafunc.parametrize("sequence_length", [metafunc.config.option.sequence_length])
    if 'batch_size' in metafunc.fixturenames:
        metafunc.parametrize("batch_size", [metafunc.config.option.batch_size])
    if 'vocab_size' in metafunc.fixturenames:
        metafunc.parametrize("vocab_size", [metafunc.config.option.vocab_size])
    if 'ctx_length' in metafunc.fixturenames:
        metafunc.parametrize("ctx_length", [metafunc.config.option.ctx_length])


@pytest.fixture(scope="session")
def sequence_length(pytestconfig):
    return pytestconfig.getoption("sequence_length")


@pytest.fixture(scope="session")
def batch_size(pytestconfig):
    return pytestconfig.getoption("batch_size")


@pytest.fixture(scope="session")
def vocab_size(pytestconfig):
    return pytestconfig.getoption("vocab_size")


@pytest.fixture(scope="session")
def ctx_length(pytestconfig):
    return pytestconfig.getoption("ctx_length")


sequence_length = None
batch_size = None
vocab_size = None
ctx_length = None


@pytest.fixture(autouse=True, scope="session")
def init(pytestconfig):

    global sequence_length
    global batch_size
    global vocab_size
    global ctx_length

    sequence_length = pytestconfig.getoption('sequence_length')
    batch_size = pytestconfig.getoption('batch_size')
    vocab_size = pytestconfig.getoption('vocab_size')
    ctx_length = pytestconfig.getoption('ctx_length')
    
    torch.set_printoptions(threshold=torch.inf)


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
    top_ks = torch.ones(batch_size, dtype=torch.int64) * vocab_size  # Disable top k    
    top_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99

    print("top_ks", top_ks)
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
    top_ks = torch.ones(batch_size, dtype=torch.int64) * vocab_size  # Disable top k    
    top_ps = torch.ones(batch_size, )  # Disable top p
    min_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99

    print("top_ks", top_ks)
    print("top_ps", top_ps)
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
def setup_data_random_sampling(batch_size, vocab_size):
    import numpy as np

    seed = np.random.randint(1, 101)
    # seed = 67
    torch.manual_seed(seed)
    
    logits = torch.randn(batch_size, 1, vocab_size)
    # temperatures = torch.randint(1, 11, (batch_size,)) / 10.0
    pseudo_random_generator = torch.Generator()
    random_numbers = torch.rand(batch_size, generator=pseudo_random_generator)
    generators = {
        i: pseudo_random_generator for i in range(batch_size)
    }
    
    return {
        "seed": seed,
        "logits": logits,
        # "temperatures": temperatures,
        "random_numbers": random_numbers,
        "generators": generators,
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
    min_ps = torch.randint(50, 100, (batch_size,)) / 100.0  # Between 0.50 and 0.99

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
        "min_ps": min_ps,
    }
