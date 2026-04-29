"""Export the trained RSL-RL actor to ONNX format for sim2sim deployment.

Usage:
    python export_onnx.py <checkpoint.pt> [output.onnx]

The training config has `empirical_normalization: false` and
`actor_obs_normalization: {}`, so we only need the actor MLP weights —
no normalizer module to fuse in.
"""

import argparse
import os

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Reconstruct the actor from RSL-RL ActorCritic checkpoint."""

    def __init__(self, obs_dim=45, act_dim=12, hidden_dims=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs):
        return self.actor(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to model_*.pt checkpoint")
    ap.add_argument("out", nargs="?", default=None, help="output .onnx path")
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)) or ".",
        "policy.onnx",
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    model = ActorNetwork(obs_dim=45, act_dim=12, hidden_dims=(512, 256, 128))

    actor_state = {k: v for k, v in state.items() if k.startswith("actor.")}
    model.load_state_dict(actor_state)
    model.eval()

    dummy_input = torch.randn(1, 45)
    torch.onnx.export(
        model,
        dummy_input,
        out,
        input_names=["obs"],
        output_names=["continuous_actions"],
        opset_version=11,
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
    )
    print(f"Exported to {out}")

    import onnxruntime as rt
    sess = rt.InferenceSession(out, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["continuous_actions"], {"obs": dummy_input.numpy()})[0]
    torch_out = model(dummy_input).detach().numpy()
    diff = abs(onnx_out - torch_out).max()
    print(f"Max diff between PyTorch and ONNX: {diff:.6e}")


if __name__ == "__main__":
    main()
