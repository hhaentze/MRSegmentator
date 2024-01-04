import argparse

import torch

parser = argparse.ArgumentParser(
    description="Separate model state dicts and optimizer state dicts."
)
parser.add_argument("--file", type=str, required=True, help="Path to the combined state dict")
parser.add_argument(
    "--model_key",
    type=str,
    required=False,
    default="network",
    help="Key under which the model weights can be accessed in the state dict",
)
parser.add_argument(
    "--optim_key",
    type=str,
    required=False,
    default="optimizer",
    help="Key under which the optimizer weights can be accessed in the state dict",
)

args = parser.parse_args()

if __name__ == "__main__":
    state_dict = torch.load(args.file)
    torch.save(state_dict["optimizer"], "optimizer.pt")
    torch.save(state_dict["network"], "model.pt")
