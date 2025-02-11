import torch


def print_difference_in_tensors(
    tensor_1, tensor_1_name, tensor_2, tensor_2_name, threshold=1e-5
):
    difference = torch.abs(tensor_1 - tensor_2)
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
        print("-"*20)
