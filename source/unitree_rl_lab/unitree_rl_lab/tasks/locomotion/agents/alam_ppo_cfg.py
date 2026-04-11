"""PPO runner config for ALaM dual-policy training (3-legged loco + manipulation).

obs_groups mapping:
    - "policy":      ["policy", "command"]  → actor base input (proprioceptive + goal/leg_state)
    - "critic":      ["policy", "command", "critic", "privileged"]  → critic full info
    - "privileged":  ["privileged"]  → encoder input (CoF vel, CoM pos)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ALaMActorCriticCfg(RslRlPpoActorCriticCfg):
    """Config for ALaM dual-policy actor-critic."""

    class_name: str = "ALaMActorCritic"

    # Manipulation actor: → [a_man_leg(3), a_man_diy(4), base_cmd(3)] = 10
    num_manip_leg_actions: int = 3
    num_manip_diy_actions: int = 4
    num_base_cmd: int = 3
    manip_actor_hidden_dims: list[int] = [256, 128]

    # Locomotion actor: → a_loc(12)
    num_loco_actions: int = 12
    loco_actor_hidden_dims: list[int] = [512, 256, 128]

    # Privileged encoder (training only)
    priv_latent_dim: int = 16
    priv_encoder_hidden_dims: list[int] = [64, 32]

    # Critic (shared)
    critic_hidden_dims: list[int] = [512, 256, 128]


@configclass
class ALaMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for ALaM pedipulation training."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False

    # Map environment observation groups to algorithm observation sets
    obs_groups = {
        "policy": ["policy", "command"],
        "critic": ["policy", "command", "critic", "privileged"],
        "privileged": ["privileged"],
    }

    policy = ALaMActorCriticCfg(
        init_noise_std=0.5,
        # Required by base class (unused by ALaMActorCritic, it uses its own dims)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
