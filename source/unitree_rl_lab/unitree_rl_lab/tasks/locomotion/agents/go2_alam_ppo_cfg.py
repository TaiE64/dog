"""PPO runner config for ALaM pedipulation on standard Go2 (no DIY arms).

Manipulation actor outputs: a_man_leg(3) + base_cmd(3) = 6 dims.
No diy joints. Total env actions = 12 (leg joints only).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .alam_ppo_cfg import ALaMActorCriticCfg


@configclass
class Go2ALaMActorCriticCfg(ALaMActorCriticCfg):
    """ALaM config for standard Go2: no diy joints."""

    num_manip_diy_actions: int = 0  # no diy arms
    manip_actor_hidden_dims: list[int] = [256, 128]
    loco_actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]


@configclass
class Go2ALaMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for ALaM on standard Go2."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "command"],
        "critic": ["policy", "command", "critic", "privileged"],
        "privileged": ["privileged"],
    }

    policy = Go2ALaMActorCriticCfg(
        init_noise_std=0.5,
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
