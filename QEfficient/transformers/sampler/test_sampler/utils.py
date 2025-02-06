import torch


def print_difference_in_tensors(
    tensor_1, tensor_1_name, tensor_2, tensor_2_name, threshold=1e-5
):
    difference = torch.abs(tensor_1 - tensor_2)
    indices = torch.nonzero(difference > threshold, as_tuple=True)

    print(f"\n\nOutput tensors {tensor_1_name} and {tensor_2_name} do not match. Details:")
    for idx in zip(*indices):
        idx_ = tuple(i.item() for i in idx)
        print(
            f"Index: {idx_}, {tensor_1_name}: {tensor_1[idx]}, {tensor_2_name}: {tensor_2[idx]}, difference: {difference[idx]}"
        )
