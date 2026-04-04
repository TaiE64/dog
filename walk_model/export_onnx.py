"""Export the trained RSL-RL actor to ONNX format for sim2sim deployment."""

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
    ckpt = torch.load("model_2900.pt", map_location="cpu")
    state = ckpt["model_state_dict"]

    model = ActorNetwork(obs_dim=45, act_dim=12, hidden_dims=(512, 256, 128))

    # Load actor weights (keys: actor.0.weight, actor.0.bias, actor.2.weight, ...)
    actor_state = {}
    for k, v in state.items():
        if k.startswith("actor."):
            actor_state[k] = v
    model.load_state_dict({"actor." + k.split("actor.")[-1]: v for k, v in actor_state.items()})

    model.eval()

    dummy_input = torch.randn(1, 45)
    torch.onnx.export(
        model,
        dummy_input,
        "go2_diyleg_walk_policy.onnx",
        input_names=["obs"],
        output_names=["continuous_actions"],
        opset_version=11,
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
    )
    print("Exported to go2_diyleg_walk_policy.onnx")

    # Verify
    import onnxruntime as rt
    sess = rt.InferenceSession("go2_diyleg_walk_policy.onnx", providers=["CPUExecutionProvider"])
    out = sess.run(["continuous_actions"], {"obs": dummy_input.numpy()})[0]
    torch_out = model(dummy_input).detach().numpy()
    diff = abs(out - torch_out).max()
    print(f"Max diff between PyTorch and ONNX: {diff:.6e}")


if __name__ == "__main__":
    main()
