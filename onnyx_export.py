#!/usr/bin/env python3
# Converts a saved PyTorch state_dict model to ONNX format.
import argparse
import os

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Export a trained PyTorch model to ONNX")
    parser.add_argument("--input", type=str, default="models/recycling_resnet18.pth", help="path to the input PyTorch weights file")
    parser.add_argument("--output", type=str, default="models/recycling_resnet18.onnx", help="output ONNX file path")
    parser.add_argument("--num-classes", type=int, default=2, help="number of classes in the trained model")
    parser.add_argument("--input-size", type=int, default=224, help="input image size expected by the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading weights from: {args.input}")
    state_dict = torch.load(args.input, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = build_model(args.num_classes).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    print(f"Exporting to ONNX: {args.output}")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Export complete: {args.output}")


if __name__ == "__main__":
    main()


