import numpy as np
import torch


def print_difference_in_tensors(
    tensor_1, tensor_1_name, tensor_2, tensor_2_name, threshold=1e-5
):
    difference = torch.abs(tensor_1 - tensor_2)
    
    # Handle the case when both tensors are scalars
    if tensor_1.numel() == 1 and tensor_2.numel() == 1:
        indices = (torch.tensor([0]),)
    else:
        indices = torch.nonzero(difference > threshold, as_tuple=True)

    # Define thresholds for different bit precisions
    thresholds = {
        "8-bit": 2**-8,
        "10-bit": 2**-10,
        "12-bit": 2**-12,
        "14-bit": 2**-14,
        "16-bit": 2**-16,
    }

    print(f"\n\nOutput tensors {tensor_1_name} and {tensor_2_name} do not match. Details:")
    for idx in zip(*indices):
        idx_ = tuple(i.item() for i in idx)
        diff = difference[idx]
        print(f"Index: {idx_}")
        print(f"{tensor_1_name}: {tensor_1[idx]}")
        print(f"{tensor_2_name}: {tensor_2[idx]}")
        print(f"Absolute Diff: {diff}")

        for bit_precision, bit_threshold in thresholds.items():
            print(f"Difference exceeds {bit_precision:<6} threshold of {bit_threshold:<18}: {diff > bit_threshold}")

        if diff <= thresholds["10-bit"]:
            print("Insignificant")
        else:
            print("Significant")
        print("-" * 20)

    max_diff = torch.max(difference)
    print(f"\nMaximum Absolute Difference: {max_diff}")

    for bit_precision, bit_threshold in thresholds.items():
        exceeds_threshold = max_diff > bit_threshold
        print(f"Maximum difference exceeds {bit_precision:<6} threshold of {bit_threshold:<18}: {exceeds_threshold}")
    print("-" * 50)


def get_float16_binary_repr(number):
    return np.binary_repr(np.float16(number).view(np.int16), width=16)


def get_summary_statistics(samples: torch.Tensor):
    mean = torch.mean(samples * 1.)
    # std = torch.std(samples * 1.)
    # min_val = torch.min(samples)
    # max_val = torch.max(samples)
    # median = torch.median(samples)
    # mode = torch.mode(samples)[0]
    variance = torch.var(samples * 1.)
    # skewness = torch.mean((samples * 1. - mean) ** 3) / (std ** 3)
    # kurtosis = torch.mean((samples * 1. - mean) ** 4) / (std ** 4) - 3
    return {
        "mean": mean.reshape((1,)),
        # "std": std.reshape((1,)),
        # "min":  min_val.reshape((1,)),
        # "max":  max_val.reshape((1,)),
        # "median": median.reshape((1,)),
        # "mode": mode.reshape((1,)),
        "variance": variance.reshape((1,)),
        # "skewness": skewness.reshape((1,)),
        # "kurtosis": kurtosis.reshape((1,)),
    }
    